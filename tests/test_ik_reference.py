"""Independent endpoint, branch, limit and authority tests for nominal IK."""

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

from nca.contracts import ContractError, LifecycleStamp, sha256_json
from nca.ik.contracts import (IKAuthority, IKObservation, RootPose,
                              target_from_observation)
from nca.ik.kinematics import PlanarChain, PlanarJoint, chain_from_profile, cpu_fk
from nca.ik.profiles import runner_right_sole_profile
from nca.ik.reference import (AnalyticTwoLink, DampedLeastSquares, SolverConfig,
                              analytic_two_link, solve_numeric)


def runner_endpoint(q):
    h, a = q
    return (3.6 - 4.95 * math.sin(h) + math.cos(h + a) - 1.25 * math.sin(h + a),
            2.8 + 4.95 * math.cos(h) + math.sin(h + a) + 1.25 * math.cos(h + a))


def two_link(length1=1.0, length2=1.0):
    return PlanarChain("analytic-fixture", "root", (
        PlanarJoint("q1", "root", "first", (0., 0.), (0., 0.), 0., -math.pi, math.pi),
        PlanarJoint("q2", "first", "second", (length1, 0.), (0., 0.), 0., -math.pi, math.pi),
    ), (length2, 0.))


def issued(profile, positions=(0.0, 0.0), **changes):
    observation = IKObservation(
        "reference-observation", 10, LifecycleStamp("reference-fixture", "runner", 1),
        profile.sha256, profile.source_sha256, profile.topology_sha256,
        RootPose((10., 20.), math.pi / 2),
        (True,) * 5 + (False,) * 3, (True,) * 4 + (False,) * 3,
        (False, False, True, True, False, False, False),
        (False, False, True, True, False, False, False),
        (None, None, *positions, None, None, None))
    return replace(observation, **changes)


def request(profile, observation, point, **changes):
    values = dict(id="reference-target", source_id="reference", goal_id="reach",
                  sequence=1, policy_sha256=sha256_json({"reference": 1}),
                  grant_id="reference-grant", grant_generation=1,
                  point=point, goal_deadline_tick=100)
    values.update(changes)
    target = target_from_observation(profile, observation, **values)
    authority = IKAuthority(observation.tick, observation.lifecycle, target.source_id,
                            target.goal_id, target.policy_sha256, target.grant_id,
                            target.grant_generation)
    return target, authority


class NumericalReferenceTests(unittest.TestCase):
    def setUp(self):
        self.profile = runner_right_sole_profile()
        self.chain = chain_from_profile(self.profile)

    def assertBounded(self, result, chain=None):
        chain = self.chain if chain is None else chain
        self.assertLessEqual(result.iterations, 64)
        self.assertLessEqual(result.maximum_step_norm, .25 + 1e-12)
        self.assertTrue(all(math.isfinite(v) for v in result.configuration))
        for q, joint in zip(result.configuration, chain.joints):
            self.assertLessEqual(joint.lower, q)
            self.assertLessEqual(q, joint.upper)

    def test_runner_full_inner_bounds_against_independent_endpoint_and_analytic_branches(self):
        for h in (-.72, -.6, -.25, 0., .25, .6, .72):
            for a in (-.27, -.2, 0., .2, .27):
                for current in ((0., 0.), (-.72, -.27), (.72, .27), (.72, -.27)):
                    with self.subTest(h=h, a=a, current=current):
                        point = runner_endpoint((h, a))
                        result = solve_numeric(self.chain, point, current,
                                               lower=(-.72, -.27), upper=(.72, .27))
                        analytic = analytic_two_link(self.chain, point, current,
                                                     lower=(-.72, -.27), upper=(.72, .27))
                        self.assertEqual(result.status, "solved")
                        self.assertEqual(analytic.status, "solved")
                        self.assertEqual(len(analytic.configurations), 1)
                        self.assertLessEqual(math.dist(runner_endpoint(result.configuration), point), 1e-6)
                        self.assertLessEqual(math.dist(result.configuration, analytic.configuration), 4e-6)
                        self.assertBounded(result)

    def test_runner_smooth_target_continuation_preserves_admitted_branch(self):
        current = (-.6, 0.)
        previous_point = runner_endpoint(current)
        for i in range(1, 121):
            expected = (-.6 + i / 100, .2 * math.sin(i * math.pi / 60))
            point = runner_endpoint(expected)
            self.assertLessEqual(math.dist(point, previous_point), .1)
            result = solve_numeric(self.chain, point, current,
                                   lower=(-.72, -.27), upper=(.72, .27))
            self.assertEqual(result.status, "solved")
            self.assertLessEqual(max(abs(a - b) for a, b in zip(result.configuration, current)), .25)
            self.assertLessEqual(math.dist(result.configuration, expected), 4e-6)
            current, previous_point = result.configuration, point

    def test_two_branches_are_selected_from_current_configuration_without_averaging(self):
        chain = two_link()
        for current, sign in (((0., .1), 1), ((1., -.1), -1)):
            result = solve_numeric(chain, (1., 1.), current)
            self.assertEqual(result.status, "solved")
            self.assertGreater(sign * result.configuration[1], 0)
            expected = ((0., math.pi / 2) if sign == 1 else (math.pi / 2, -math.pi / 2))
            self.assertLess(math.dist(result.configuration, expected), 1e-6)
            self.assertBounded(result, chain)

    def test_synthetic_branch_continuation_does_not_flip_elbow(self):
        chain = two_link()
        bend = 2 * math.acos(.75)
        current = (-1.0 - bend / 2, bend)
        for i in range(61):
            angle = -1.0 + i / 30
            point = 1.5 * math.cos(angle), 1.5 * math.sin(angle)
            result = solve_numeric(chain, point, current)
            self.assertEqual(result.status, "solved")
            self.assertGreater(result.configuration[1], 0)
            self.assertLess(max(abs(a - b) for a, b in zip(result.configuration, current)), .25)
            current = result.configuration

    def test_straight_and_exactly_folded_singular_starts_escape_with_bounded_steps(self):
        chain = two_link()
        for point, current in (((1.5, 0.), (0., 0.)), ((.1, 0.), (0., math.pi))):
            with self.subTest(point=point, current=current):
                result = solve_numeric(chain, point, current)
                self.assertEqual(result.status, "solved")
                q1, q2 = result.configuration
                independent = math.cos(q1) + math.cos(q1 + q2), math.sin(q1) + math.sin(q1 + q2)
                self.assertLessEqual(math.dist(independent, point), 1e-6)
                self.assertBounded(result, chain)

    def test_near_singular_and_target_at_singularity_have_finite_behavior(self):
        chain = two_link()
        result = solve_numeric(chain, (2., 0.), (-.01, .02))
        self.assertEqual(result.status, "solved")
        self.assertBounded(result, chain)
        self.assertLessEqual(result.residual_cells, 1e-6)

    def test_joint_limited_target_is_best_effort_without_geometric_claim(self):
        point = runner_endpoint((1.1, 0.))
        result = solve_numeric(self.chain, point, (0., 0.), lower=(-.72, -.27), upper=(.72, .27))
        self.assertEqual(result.status, "best_effort")
        self.assertGreater(result.residual_cells, 1e-6)
        self.assertTrue(result.active_limits)
        self.assertIn("unknown", result.reason)
        self.assertBounded(result)

    def test_numerical_stagnation_alone_does_not_certify_unreachable(self):
        result = solve_numeric(two_link(), (3., 0.), (0., 0.))
        self.assertEqual(result.status, "best_effort")
        self.assertEqual(result.point, (2., 0.))
        self.assertEqual(result.residual_cells, 1.)

    def test_iteration_and_time_limits_do_not_report_infeasibility_or_action(self):
        for config, status in ((SolverConfig(max_iterations=0), "iteration_budget_exhausted"),
                               (SolverConfig(max_seconds=0), "time_budget_exhausted"),
                               (SolverConfig(max_iterations=1), "iteration_budget_exhausted")):
            with self.subTest(config=config):
                result = solve_numeric(self.chain, runner_endpoint((.6, .2)), (0., 0.), config=config)
                self.assertEqual(result.status, status)
                self.assertFalse(result.actionable)
                self.assertLessEqual(result.iterations, config.max_iterations)

    def test_wall_budget_is_checked_during_solver_not_only_at_entry(self):
        with patch("nca.ik.reference.time.perf_counter", side_effect=[0., 0., 0., 0., .1, .1]):
            result = solve_numeric(self.chain, runner_endpoint((.6, .2)), (0., 0.))
        self.assertEqual(result.status, "time_budget_exhausted")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.evaluations, 1)
        self.assertIsNotNone(result.configuration)

    def test_initial_projection_is_distinct_from_iterative_correction(self):
        result = solve_numeric(self.chain, runner_endpoint((.3, .1)), (.78, .31),
                               lower=(-.72, -.27), upper=(.72, .27))
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.projected_coordinates, (0, 1))
        self.assertGreaterEqual(result.projection_events, 1)
        self.assertGreaterEqual(result.projection_correction_norm, math.hypot(.06, .04) - 1e-12)
        self.assertGreater(result.correction_norm, 0)

    def test_angle_equivalence_does_not_cross_hard_stops(self):
        joint = PlanarJoint("rotary", "root", "tip", (0., 0.), (0., 0.), 0., -math.pi, math.pi)
        chain = PlanarChain("wrap-fixture", "root", (joint,), (1., 0.))
        point = math.cos(-3.0), math.sin(-3.0)
        result = solve_numeric(chain, point, (3.,), config=SolverConfig(max_iterations=1))
        self.assertNotEqual(result.status, "solved")
        self.assertLessEqual(abs(result.configuration[0] - 3.), .25 + 1e-12)
        wrapped = solve_numeric(chain, (1., 0.), (math.tau,))
        self.assertEqual(wrapped.status, "solved")
        self.assertEqual(wrapped.configuration, (0.,))
        self.assertEqual(wrapped.projected_coordinates, (0,))

    def test_masks_preserve_passive_geometry_and_refuse_missing_path_state(self):
        chain = replace(two_link(), joints=(replace(two_link().joints[0], participation="passive"),
                                           two_link().joints[1]))
        target = math.cos(.3) + math.cos(.7), math.sin(.3) + math.sin(.7)
        solved = solve_numeric(chain, target, (.3, 0.))
        self.assertEqual(solved.status, "solved")
        self.assertEqual(solved.configuration[0], .3)
        self.assertEqual(solve_numeric(chain, target, (.3, 0.), active=(True, True)).status, "invalid_input")
        self.assertEqual(solve_numeric(chain, target, (None, 0.), measured=(False, True)).status,
                         "missing_required_measurement")
        self.assertEqual(solve_numeric(chain, target, (.3, 0.), present=(False, True)).status, "invalid_input")
        inactive = solve_numeric(self.chain, runner_endpoint((.6, .2)), (0., 0.), active=(False, False))
        self.assertEqual(inactive.status, "best_effort")
        self.assertEqual(inactive.configuration, (0., 0.))

    def test_task_masks_are_explicit_and_empty_tasks_are_invalid(self):
        chain = PlanarChain("slide", "root", (
            PlanarJoint("slide", "root", "tip", (0., 0.), (0., 0.), 0., -2., 2., kind="prismatic"),
        ), (0., 0.))
        result = solve_numeric(chain, (1., 100.), (0.,), task_mask=(True, False))
        self.assertEqual(result.status, "solved")
        self.assertLessEqual(abs(result.configuration[0] - 1.), 1e-6)
        self.assertEqual(result.point[1], 0.)
        self.assertEqual(solve_numeric(chain, (1., 0.), (0.,), task_mask=(False, False)).status, "invalid_input")

    def test_mixed_numerical_fixture_is_solved_without_a_physical_actuator_claim(self):
        chain = PlanarChain("mixed-fixture", "root", (
            PlanarJoint("angle", "root", "arm", (0., 0.), (0., 0.), 0., -1., 1.),
            PlanarJoint("slide", "arm", "tip", (1., 0.), (0., 0.), 0., 0., 2., kind="prismatic"),
        ), (0., 0.))
        target = 1.6 * math.cos(.3), 1.6 * math.sin(.3)
        result = solve_numeric(chain, target, (0., 0.))
        self.assertEqual(result.status, "solved")
        self.assertLessEqual(math.dist(result.configuration, (.3, .6)), 1e-6)
        self.assertBounded(result, chain)

    def test_posture_preference_biases_redundancy_without_replacing_the_point_task(self):
        chain = PlanarChain("redundant-slides", "root", (
            PlanarJoint("first", "root", "a", (0., 0.), (0., 0.), 0., 0., 2., kind="prismatic"),
            PlanarJoint("second", "a", "b", (0., 0.), (0., 0.), 0., 0., 2., kind="prismatic"),
        ), (0., 0.))
        result = solve_numeric(chain, (1., 0.), (0., 0.), posture=(.8, .2),
                               config=SolverConfig(posture_weight=.1))
        self.assertEqual(result.status, "solved")
        self.assertLessEqual(abs(sum(result.configuration) - 1.), 1e-6)
        self.assertGreater(result.configuration[0], result.configuration[1])

    def test_invalid_values_and_configurations_never_make_nonfinite_payloads(self):
        for target, current, extra in (((math.nan, 0.), (0., 0.), {}),
                                       ((0., 0.), (math.inf, 0.), {}),
                                       ((0., 0.), (True, 0.), {}),
                                       ((0., 0.), (0.,), {}),
                                       ((0., 0.), (0., 0.), {"active": (1, 1)}),
                                       ((0., 0.), (0., 0.), {"lower": (-1., -1.)})):
            with self.subTest(target=target, current=current, extra=extra):
                result = solve_numeric(self.chain, target, current, **extra)
                self.assertEqual(result.status, "invalid_input")
                self.assertIsNone(result.configuration)
        for changes in ({"max_iterations": 257}, {"max_seconds": math.inf},
                        {"damping": 0}, {"max_step": 1}, {"max_backtracks": 0},
                        {"tolerance_cells": 0}, {"posture_weight": .2}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                SolverConfig(**changes)


class AnalyticReferenceTests(unittest.TestCase):
    def test_independent_two_link_answers_and_branch_order(self):
        chain = two_link()
        plus = analytic_two_link(chain, (1., 1.), (0., 1.))
        minus = analytic_two_link(chain, (1., 1.), (1., -1.))
        self.assertEqual(plus.status, "solved")
        self.assertEqual(len(plus.configurations), 2)
        self.assertLess(math.dist(plus.configuration, (0., math.pi / 2)), 1e-12)
        self.assertLess(math.dist(minus.configuration, (math.pi / 2, -math.pi / 2)), 1e-12)
        for point in plus.points:
            self.assertLess(math.dist(point, (1., 1.)), 1e-12)

    def test_analytic_path_does_not_call_numeric_fk_or_solver(self):
        with patch("nca.ik.reference.cpu_fk", side_effect=AssertionError("shared FK used")), \
             patch("nca.ik.reference.solve_numeric", side_effect=AssertionError("numeric used")):
            result = AnalyticTwoLink().solve(two_link(), (1., 1.), (0., 0.))
        self.assertEqual(result.status, "solved")

    def test_displaced_anchors_and_nonzero_authored_angles(self):
        # Effective vectors are (2,0), (1,0), about displaced root pivot (3,4).
        # Joint zeros +pi/2 and -pi/2 are cancelled only by the authored vectors.
        chain = PlanarChain("angled-anchors", "root", (
            PlanarJoint("first", "root", "a", (3., 4.), (.5, .25), math.pi / 2, -math.pi, math.pi),
            PlanarJoint("second", "a", "b", (.5, -1.75), (-.2, .4), -math.pi / 2, -math.pi, math.pi),
        ), (.8, .4))
        result = analytic_two_link(chain, (5., 5.), (0., 1.))
        self.assertEqual(result.status, "solved")
        for point in result.points:
            self.assertLess(math.dist(point, (5., 5.)), 1e-12)
        self.assertLess(math.dist(result.configuration, (0., math.pi / 2)), 1e-12)

    def test_annular_hole_and_outer_circle_are_distinct_geometric_proofs(self):
        for point, proof in (((4., 0.), "outside_outer_radius"), ((.5, 0.), "inside_inner_radius")):
            result = analytic_two_link(two_link(2., 1.), point, (0., 0.))
            self.assertEqual(result.status, "geometrically_infeasible")
            self.assertEqual(result.proof_kind, proof)
            self.assertEqual((result.inner_radius, result.outer_radius), (1., 3.))
            self.assertFalse(result.configurations)

    def test_joint_limits_filter_legal_branch_not_average_or_clamp(self):
        chain = two_link()
        result = analytic_two_link(chain, (1., 1.), (0., 0.), lower=(-.1, -.1), upper=(.1, .1))
        self.assertEqual(result.status, "joint_limited")
        self.assertFalse(result.configurations)
        self.assertIsNone(result.proof_kind)

    def test_folded_continuum_retains_current_first_angle(self):
        result = analytic_two_link(two_link(), (0., 0.), (.7, math.pi))
        self.assertEqual(result.status, "solved")
        self.assertAlmostEqual(result.configuration[0], .7)
        self.assertAlmostEqual(result.configuration[1], math.pi)
        self.assertLess(math.hypot(*result.points[0]), 1e-12)

    def test_folded_continuum_projects_current_first_angle_into_narrow_solve_bounds(self):
        result = analytic_two_link(two_link(), (0., 0.), (1., math.pi),
                                   lower=(-.5, -math.pi), upper=(.5, math.pi))
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.configuration[0], .5)
        self.assertLess(math.hypot(*result.points[0]), 1e-12)

    def test_branch_order_uses_equivalent_bounded_current_configuration(self):
        q1, q2 = -2.9, 1.
        point = math.cos(q1) + math.cos(q1 + q2), math.sin(q1) + math.sin(q1 + q2)
        canonical = analytic_two_link(two_link(), point, (q1, q2))
        wrapped = analytic_two_link(two_link(), point, (q1 + math.tau, q2))
        self.assertEqual(canonical.configurations, wrapped.configurations)
        self.assertLess(math.dist(wrapped.configuration, (q1, q2)), 1e-12)

    def test_non_rotary_and_degenerate_analytic_mechanisms_are_explicitly_unsupported(self):
        chain = two_link()
        slide = replace(chain, joints=(replace(chain.joints[0], kind="prismatic"), chain.joints[1]))
        for value in (slide, two_link(0., 1.), None):
            self.assertEqual(analytic_two_link(value, (1., 1.), (0., 0.)).status, "unsupported_profile")


class SourceBoundReferenceTests(unittest.TestCase):
    def setUp(self):
        self.profile = runner_right_sole_profile()
        self.observation = issued(self.profile)
        self.solver = DampedLeastSquares(profile=self.profile)

    def test_solved_result_binds_target_and_independently_validates(self):
        goal, authority = request(self.profile, self.observation, runner_endpoint((.3, .15)))
        result, diagnostic = self.solver.solve_with_diagnostics(self.profile, goal, self.observation, authority)
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.target_sha256, goal.sha256)
        self.assertEqual(result.iterations, diagnostic.iterations)
        result.validate_against(self.profile, goal, self.observation, authority)
        self.assertLess(math.dist(runner_endpoint(result.reference.values), goal.point_root), 1e-6)

    def test_world_target_uses_the_same_issued_root(self):
        local = runner_endpoint((.3, -.1))
        world = 10. - local[1], 20. + local[0]
        goal, authority = request(self.profile, self.observation, world, input_frame="world")
        result = self.solver.solve(self.profile, goal, self.observation, authority)
        self.assertEqual(result.status, "solved")
        self.assertLess(math.dist(result.reference.values, (.3, -.1)), 4e-6)
        changed = replace(self.observation, root_pose=RootPose((11., 20.), math.pi / 2))
        self.assertEqual(self.solver.solve(self.profile, goal, changed, authority).status, "invalid_input")

    def test_bounded_delta_is_a_best_effort_step_not_a_falsely_solved_full_target(self):
        goal, authority = request(self.profile, self.observation, runner_endpoint((.6, .2)), output_kind="joint_delta")
        result = self.solver.solve(self.profile, goal, self.observation, authority)
        self.assertEqual(result.status, "best_effort")
        self.assertEqual(result.reference.kind, "joint_delta")
        self.assertLessEqual(max(abs(v) for v in result.reference.values), .25)
        result.validate_against(self.profile, goal, self.observation, authority)

    def test_delta_subtraction_rounding_never_exceeds_exact_quarter_radian_bound(self):
        for hip in (-.3, .3):
            for ankle_target in (-.27, .27):
                observation = issued(self.profile, (hip, 0.))
                point = runner_endpoint((math.copysign(.72, hip), ankle_target))
                goal, authority = request(self.profile, observation, point, output_kind="joint_delta")
                result, diagnostic = self.solver.solve_with_diagnostics(self.profile, goal, observation, authority)
                self.assertEqual(result.status, "best_effort")
                self.assertLessEqual(max(abs(v) for v in result.reference.values), .25)
                resolved = tuple(old + delta for old, delta in zip((hip, 0.), result.reference.values))
                self.assertEqual(diagnostic.configuration, resolved)
                self.assertIn(0, diagnostic.emission_adjusted_coordinates)
                result.validate_against(self.profile, goal, observation, authority)

    def test_compliant_current_projection_and_impossible_delta_envelope_are_separate(self):
        for positions, mode, status in (((.78, .31), "joint_position", "solved"),
                                         ((2., 0.), "joint_delta", "invalid_input")):
            observation = issued(self.profile, positions)
            goal, authority = request(self.profile, observation, runner_endpoint((.3, .1)), output_kind=mode)
            result = self.solver.solve(self.profile, goal, observation, authority)
            self.assertEqual(result.status, status)
            if status == "solved":
                result.validate_against(self.profile, goal, observation, authority)
            else:
                self.assertIsNone(result.reference)

    def test_outer_radius_proof_is_checked_without_conflating_joint_limited_targets(self):
        goal, authority = request(self.profile, self.observation, (20., 20.))
        result = self.solver.solve(self.profile, goal, self.observation, authority)
        self.assertEqual(result.status, "geometrically_infeasible")
        self.assertEqual(result.proof.kind, "outside_outer_radius")
        self.assertIsNone(result.reference)
        self.assertEqual(result.iterations, 0)
        result.validate_against(self.profile, goal, self.observation, authority)
        limited, authority = request(self.profile, self.observation, runner_endpoint((1.1, 0.)))
        self.assertEqual(self.solver.solve(self.profile, limited, self.observation, authority).status, "best_effort")

    def test_timeout_and_iteration_exhaustion_never_emit_reference(self):
        goal, authority = request(self.profile, self.observation, runner_endpoint((.6, .2)))
        for config in (SolverConfig(max_iterations=0), SolverConfig(max_seconds=0)):
            result, diagnostic = DampedLeastSquares(config=config).solve_with_diagnostics(
                self.profile, goal, self.observation, authority)
            self.assertEqual(result.status, "iteration_budget_exhausted")
            self.assertIsNone(result.reference)
            self.assertIsNotNone(diagnostic)
            self.assertFalse(diagnostic.actionable)

    def test_stale_cancelled_replayed_missing_or_changed_identity_have_no_payload(self):
        goal, authority = request(self.profile, self.observation, runner_endpoint((.3, .1)))
        for changed in (replace(authority, current_tick=22), replace(authority, revoked=True),
                        replace(authority, last_consumed_sequence=1),
                        replace(authority, grant_generation=2)):
            result = self.solver.solve(self.profile, goal, self.observation, changed)
            self.assertEqual(result.status, "invalid_input")
            self.assertIsNone(result.reference)
        missing = replace(self.observation, root_pose=None)
        self.assertEqual(self.solver.solve(self.profile, goal, missing, authority).status, "missing_required_measurement")
        changed = replace(self.profile, authored_play_radians=.06)
        self.assertEqual(self.solver.solve(changed, goal, self.observation, authority).status, "unsupported_profile")


if __name__ == "__main__":
    unittest.main()
