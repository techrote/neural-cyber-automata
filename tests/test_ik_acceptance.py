"""Acceptance/profile binding and dimensional checks; no learner or host run."""

import hashlib
import math
from pathlib import Path
import unittest

from nca.contracts import strict_json_loads
from nca.ik.profiles import IKProfile, nominal_pose, runner_right_sole_profile


ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE = ROOT / "configs/ik/runner_right_sole_acceptance_v1.json"


class IKAcceptanceDeclarationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.declaration = strict_json_loads(ACCEPTANCE.read_bytes())
        cls.binding = cls.declaration["profile_binding"]
        cls.descriptor_bytes = (ROOT / cls.binding["descriptor_path"]).read_bytes()
        cls.profile = IKProfile.from_json(cls.descriptor_bytes)

    def test_declaration_binds_packaged_geometry_in_both_hash_spaces(self):
        self.assertEqual(self.binding["descriptor_file_sha256"],
                         hashlib.sha256(self.descriptor_bytes).hexdigest())
        self.assertEqual(self.binding["profile_semantic_sha256"], self.profile.sha256)
        self.assertEqual(self.binding["profile_id"], self.profile.id)
        self.assertEqual(self.binding["profile_version"], self.profile.version)
        self.assertEqual(runner_right_sole_profile(), self.profile)
        self.assertEqual(self.declaration["source"]["revision"], self.profile.source.commit)
        self.assertEqual(self.declaration["source"]["repository"], self.profile.source.repository)

    def test_sole_scale_and_play_allowance_match_the_descriptor(self):
        geometry = self.declaration["geometry"]
        self.assertEqual(tuple(geometry["effector_point_local_cells"]),
                         self.profile.effector.point_local)
        self.assertEqual(geometry["effector_part_id"], self.profile.effector.part_id)
        self.assertAlmostEqual(geometry["nominal_reach_cells"],
                               self.profile.nominal_reach_cells, places=12)
        first, second = self.profile.effector_chain
        link_one = math.dist(first.child_anchor_local, second.parent_anchor_local)
        link_two = math.dist(second.child_anchor_local, self.profile.effector.point_local)
        self.assertAlmostEqual(link_one, math.hypot(*geometry["hip_to_ankle_rest_vector_cells"]),
                               places=12)
        self.assertAlmostEqual(link_two, math.hypot(*geometry["ankle_to_effector_rest_vector_cells"]),
                               places=12)
        self.assertEqual(geometry["authored_play_radians"], self.profile.authored_play_radians)
        allowance = self.profile.authored_play_radians * (link_one + 2 * link_two)
        self.assertAlmostEqual(geometry["play_only_first_order_endpoint_allowance_cells"],
                               allowance, places=12)
        neutral = nominal_pose(self.profile, dict.fromkeys(self.profile.solver_joint_ids, 0.0))
        independent_neutral = tuple(sum(parts) for parts in zip(
            geometry["root_hip_anchor_cells"], geometry["hip_to_ankle_rest_vector_cells"],
            geometry["ankle_to_effector_rest_vector_cells"]))
        for actual, expected in zip(neutral.point, independent_neutral):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_bounds_workspace_and_cadence_do_not_change_host_authority(self):
        geometry, units = self.declaration["geometry"], self.declaration["units"]
        self.assertFalse(geometry["root_control"])
        self.assertFalse(geometry["orientation_objective"])
        self.assertFalse(geometry["physical_workspace_certified"])
        self.assertEqual(units["decision_ticks"], self.profile.decision_ticks)
        self.assertEqual(units["maximum_observation_age_ticks"],
                         self.profile.max_observation_age_ticks)
        self.assertEqual(units["maximum_target_lease_ticks"], self.profile.max_target_lease_ticks)
        self.assertLessEqual(units["maximum_helper_lease_ticks"],
                             self.profile.max_observation_age_ticks)
        selected = set(self.profile.solver_joint_ids)
        self.assertEqual(set(geometry["hard_joint_bounds_radians"]), selected)
        for joint, workspace_lower, workspace_upper in zip(
                self.profile.effector_chain, self.profile.workspace.lower, self.profile.workspace.upper):
            with self.subTest(joint=joint.id):
                self.assertEqual(geometry["hard_joint_bounds_radians"][joint.id],
                                 [joint.lower, joint.upper])
                self.assertEqual(geometry["inner_target_bounds_radians"][joint.id],
                                 [joint.target_lower, joint.target_upper])
                self.assertEqual(geometry["nominal_workspace_bounds_radians"][joint.id],
                                 [workspace_lower, workspace_upper])
                self.assertAlmostEqual(joint.target_lower - joint.lower,
                                       geometry["inner_target_margin_radians"], places=12)
                self.assertAlmostEqual(joint.upper - joint.target_upper,
                                       geometry["inner_target_margin_radians"], places=12)

    def test_continuity_bound_has_geometric_headroom_in_declared_domain(self):
        geometry = self.declaration["geometry"]
        continuity = self.declaration["metrics"]["continuity"]
        a_y = geometry["hip_to_ankle_rest_vector_cells"][1]
        b_x, b_y = geometry["ankle_to_effector_rest_vector_cells"]
        ankle_max = geometry["inner_target_bounds_radians"]["ankle_1"][1]
        determinant_min = a_y * (b_x * math.cos(ankle_max) - b_y * math.sin(ankle_max))
        self.assertGreater(determinant_min, 0)
        self.assertAlmostEqual(determinant_min,
                               geometry["inner_domain_jacobian_determinant_min_cells_squared"],
                               places=12)
        point_step = continuity["qualifying_target_step_max"]["value"]
        ankle_step_bound = geometry["nominal_reach_cells"] * point_step / determinant_min
        self.assertLess(ankle_step_bound, continuity["joint_change_max"]["value"])
        self.assertEqual(continuity["joint_change_max"]["value"], self.profile.max_delta_radians)
        # Conservative angle/radius estimates also cover the hip on this same branch.
        minimum_radius = a_y - math.hypot(b_x, b_y)
        hip_step_bound = (2 * math.asin(point_step / (2 * minimum_radius))
                          + math.hypot(b_x, b_y) / minimum_radius * ankle_step_bound)
        self.assertLess(hip_step_bound, continuity["joint_change_max"]["value"])

    def test_sample_minima_cover_bins_partitions_and_every_training_seed(self):
        populations = self.declaration["populations"]
        seeds = populations["training_seeds"]
        self.assertEqual(len(seeds), 3)
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertTrue(all(type(seed) is int for seed in seeds))
        fractions = populations["split_fraction"]
        self.assertEqual(set(fractions), {"training", "tuning", "sealed_evaluation"})
        self.assertAlmostEqual(sum(fractions.values()), 1)
        self.assertTrue(all(0 < value < 1 for value in fractions.values()))
        for fraction in fractions.values():
            count = populations["minimum_generation_families"] * fraction
            self.assertAlmostEqual(count, round(count))
        nominal = populations["geometric_nominal"]
        grid = nominal["coverage_grid"]
        grid_minimum = grid["hip_bins"] * grid["ankle_bins"] * grid["minimum_queries_per_bin"]
        self.assertGreaterEqual(nominal["minimum_point_queries_per_seed"], grid_minimum)
        self.assertLessEqual(nominal["minimum_holdout_region_queries_per_seed"],
                             nominal["minimum_point_queries_per_seed"])
        physical = populations["physical"]
        self.assertGreaterEqual(physical["minimum_nominal_episodes_per_bundle"],
                                physical["minimum_static_episodes_per_bundle"]
                                + physical["minimum_sweep_episodes_per_bundle"])
        for name in ("geometric_stress", "lifecycle"):
            classes = populations[name]["classes"]
            self.assertEqual(len(classes), len(set(classes)))
            self.assertGreater(populations[name]["minimum_cases_per_class"], 0)

    def test_physical_tolerances_allow_authored_play_and_keep_numeric_gates_distinct(self):
        metrics, geometry = self.declaration["metrics"], self.declaration["geometry"]
        physical = metrics["physical"]
        self.assertGreater(physical["static_endpoint_max"]["cells"],
                           geometry["play_only_first_order_endpoint_allowance_cells"])
        self.assertLess(metrics["geometry_reference"]["fixture_endpoint_max"]["value"],
                        metrics["export_parity"]["fk_endpoint_difference_max"]["value"])
        self.assertLess(metrics["export_parity"]["fk_endpoint_difference_max"]["value"],
                        metrics["learned_raw"]["endpoint_p95_max"]["cells"])
        self.assertLess(metrics["learned_raw"]["endpoint_worst_max"]["cells"],
                        physical["static_endpoint_max"]["cells"])
        self.assertLessEqual(physical["settling_dwell_min"]["value"],
                             physical["settling_deadline"]["value"])
        for section, field in (("learned_raw", "endpoint_p95_max"),
                               ("learned_raw", "endpoint_worst_max"),
                               ("correction", "corrected_endpoint_p95_max"),
                               ("correction", "corrected_endpoint_worst_max"),
                               ("physical", "static_endpoint_max"),
                               ("physical", "sweep_endpoint_p95_max"),
                               ("physical", "sweep_endpoint_worst_max")):
            with self.subTest(section=section, field=field):
                limit = metrics[section][field]
                self.assertGreater(limit["cells"], 0)
                self.assertGreater(limit["normalized_reach"], 0)
                self.assertLess(limit["normalized_reach"], 1)
        self.assertTrue(self.declaration["units"]["conjunctive_limits"])

    def test_physical_suite_requires_motion_and_cannot_qualify_an_initial_hold(self):
        physical = self.declaration["metrics"]["physical"]
        counts = self.declaration["populations"]["physical"]
        reach = self.profile.nominal_reach_cells
        settling_envelope = min(physical["static_endpoint_max"]["cells"],
                                reach * physical["static_endpoint_max"]["normalized_reach"])
        self.assertGreater(physical["static_step_fixture_min"]["value"], settling_envelope)
        self.assertLessEqual(physical["static_step_fixture_min"]["value"],
                             physical["static_step_fixture_max"]["value"])
        self.assertGreater(counts["minimum_directed_response_trials_per_bundle"], 0)
        self.assertLessEqual(counts["minimum_directed_response_trials_per_bundle"],
                             counts["minimum_static_episodes_per_bundle"])
        self.assertGreater(physical["response_requested_joint_change_min"]["value"],
                           self.profile.authored_play_radians)
        self.assertGreater(physical["response_requested_joint_change_min"]["value"],
                           physical["motion_response_threshold"]["joint_radians"])
        tracking_envelope = min(physical["sweep_endpoint_p95_max"]["cells"],
                                reach * physical["sweep_endpoint_p95_max"]["normalized_reach"])
        diameter = physical["sweep_target_diameter_min"]["value"]
        self.assertGreater(diameter, 2 * tracking_envelope)
        self.assertGreaterEqual(physical["sweep_target_path_length_min"]["value"], 2 * diameter)
        self.assertGreater(physical["sweep_target_moving_speed_min"]["value"], 0)
        self.assertGreater(physical["sweep_target_moving_fraction_min"]["value"], 0)
        self.assertLessEqual(physical["sweep_target_moving_fraction_min"]["value"], 1)
        self.assertGreater(physical["sweep_target_reversals_min"]["value"], 0)
        seconds = counts["minimum_sweep_scored_ticks"] / self.declaration["units"]["base_hz"]
        self.assertLessEqual(physical["sweep_target_path_length_min"]["value"],
                             physical["sweep_target_speed_max"]["value"] * seconds)
        hip_lower, hip_upper = self.profile.workspace.lower[0], self.profile.workspace.upper[0]
        endpoint_a = nominal_pose(self.profile, {"hip_1": hip_lower, "ankle_1": 0}).point
        endpoint_b = nominal_pose(self.profile, {"hip_1": hip_upper, "ankle_1": 0}).point
        # A geometric capacity check only; it does not certify a clear physical path.
        self.assertLess(diameter, math.dist(endpoint_a, endpoint_b))
        control = physical["stationary_reference_control"]
        self.assertTrue(control["required"])
        self.assertLess(control["success_fraction_max"], physical["nominal_success_fraction_min"]["value"])
        self.assertGreater(control["learned_success_fraction_gain_min"], 0)
        self.assertLessEqual(control["success_fraction_max"] + control["learned_success_fraction_gain_min"], 1)
        self.assertEqual(physical["settling_deadline"]["boundary"], "completed_dwell_end_inclusive")


if __name__ == "__main__":
    unittest.main()
