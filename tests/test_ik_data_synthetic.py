"""Opened geometric dataset fixtures; never Runner singular/physical evidence."""

from pathlib import Path
from types import SimpleNamespace
import math
import unittest

from nca.contracts import ContractError, sha256_json, strict_json_loads
from nca.ik.contracts import RootPose, point_to_root
from nca.ik.kinematics import PlanarChain, PlanarJoint, chain_from_profile, cpu_fk
from nca.ik.profiles import runner_right_sole_profile
from nca.ik.reference import SolverConfig, analytic_two_link, solve_numeric


FIXTURE = Path(__file__).parent / "fixtures" / "ik" / "dataset-independent-v1.json"


def independent_equal_links(q):
    first, second = q
    return (math.cos(first) + math.cos(first + second),
            math.sin(first) + math.sin(first + second))


class IndependentDatasetFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = strict_json_loads(FIXTURE.read_bytes())
        cls.profile = runner_right_sole_profile()
        cls.synthetic = PlanarChain("dataset-fixture-equal-two-link", "root", (
            PlanarJoint("first", "root", "arm", (0., 0.), (0., 0.),
                        0., -math.pi, math.pi),
            PlanarJoint("second", "arm", "tip", (1., 0.), (0., 0.),
                        0., -math.pi, math.pi),
        ), (1., 0.))

    def test_fixture_scope_and_original_runner_identity_are_explicit(self):
        self.assertEqual(self.fixture["evidence_kind"], "opened_development_conformance_only")
        self.assertFalse(self.fixture["physical_episodes"])
        self.assertFalse(self.fixture["sealed_qualification_coverage"])
        self.assertFalse(self.fixture["synthetic_two_link"]["runner_coverage"])
        self.assertEqual(self.fixture["runner"]["profile_sha256"], self.profile.sha256)
        self.assertEqual(self.fixture["runner"]["only_admitted_branch"], "negative")

    def test_runner_targets_match_hand_values_and_keep_three_bound_domains(self):
        runner = self.fixture["runner"]
        chain = chain_from_profile(self.profile)
        for row in runner["poses"]:
            with self.subTest(q=row["q"]):
                point = cpu_fk(chain, row["q"]).point
                self.assertLess(math.dist(point, row["point"]), 1e-12)
        for index, joint in enumerate(self.profile.effector_chain):
            self.assertEqual(runner["hard_bounds"][index], [joint.lower, joint.upper])
            self.assertEqual(runner["command_bounds"][index],
                             [joint.target_lower, joint.target_upper])
        self.assertEqual(runner["nominal_workspace"], [[-.6, .6], [-.2, .2]])
        for ankle in (-.27, -.1, 0., .1, .27):
            bend = ankle + math.atan2(1.25, 1.) - math.pi / 2
            self.assertLess(math.sin(bend), 0.)
            self.assertGreater(abs(4.95 * math.hypot(1., 1.25) * math.sin(bend)), 3.12)

    def test_world_frame_fixture_uses_issued_root(self):
        frame = self.fixture["runner"]["frame"]
        observation = SimpleNamespace(root_pose=RootPose(
            tuple(frame["root_position"]), frame["root_angle"]))
        converted = point_to_root(tuple(frame["point_world"]), "world", observation)
        self.assertLess(math.dist(converted, frame["point_root"]), 1e-12)
        moved = SimpleNamespace(root_pose=RootPose((11., 20.), frame["root_angle"]))
        self.assertGreater(math.dist(point_to_root(tuple(frame["point_world"]), "world", moved),
                                     frame["point_root"]), .9)

    def test_same_endpoint_conditioned_on_distinct_postures_has_distinct_labels(self):
        rows = self.fixture["synthetic_two_link"]["cases"][:2]
        self.assertEqual(rows[0]["target"], rows[1]["target"])
        permitted_keys = ("target", "current", "posture")
        fingerprints = [sha256_json({key: row[key] for key in permitted_keys}) for row in rows]
        self.assertNotEqual(*fingerprints)
        solutions = []
        for row in rows:
            result = solve_numeric(self.synthetic, row["target"], row["current"],
                                   posture=row["posture"], config=SolverConfig(max_seconds=1.))
            self.assertEqual(result.status, row["numeric_status"])
            self.assertLess(math.dist(result.configuration, row["expected_q"]), 1e-6)
            self.assertLess(math.dist(independent_equal_links(result.configuration), row["target"]), 1e-6)
            solutions.append(result.configuration)
        averaged_pose = tuple((left + right) / 2 for left, right in zip(*solutions))
        self.assertGreater(math.dist(independent_equal_links(averaged_pose), rows[0]["target"]), .5)

    def test_synthetic_singular_and_near_singular_rows_are_separately_labelled(self):
        for row in self.fixture["synthetic_two_link"]["cases"][2:5]:
            with self.subTest(case=row["id"]):
                result = solve_numeric(self.synthetic, row["target"], row["current"],
                                       posture=row["posture"], config=SolverConfig(max_seconds=1.))
                self.assertEqual(result.status, "solved")
                self.assertLess(math.dist(independent_equal_links(result.configuration), row["target"]), 1e-6)
                bend_sine = math.sin(row["expected_q"][1])
                if row["branch"] == "singular":
                    self.assertLess(abs(bend_sine), 1e-12)
                else:
                    self.assertTrue(0 < bend_sine < .002)

    def test_raw_failure_and_independent_feasibility_are_distinct_labels(self):
        rows = self.fixture["synthetic_two_link"]["cases"][5:]
        self.assertEqual(len(rows), 2)
        for row in rows:
            with self.subTest(case=row["id"]):
                numeric = solve_numeric(self.synthetic, row["target"], row["current"],
                    posture=row["posture"],
                    config=SolverConfig(max_seconds=1., max_iterations=row.get("max_iterations", 64)))
                analytic = analytic_two_link(self.synthetic, row["target"], row["current"])
                self.assertEqual(numeric.status, row["numeric_status"])
                self.assertEqual(analytic.status, row["analytic_status"])
                self.assertFalse(row["pose_loss"])
                self.assertNotEqual(numeric.status, analytic.status)
                self.assertIsNotNone(numeric.configuration)
                if row["id"] == "iteration-exhausted":
                    self.assertFalse(numeric.actionable)
                    self.assertEqual(numeric.iterations, 0)

    def test_invalid_json_specimens_are_preserved_without_zero_substitution(self):
        for row in self.fixture["malformed_specimens"]:
            with self.subTest(case=row["id"]), self.assertRaises(ContractError) as caught:
                strict_json_loads(row["text"])
            self.assertEqual(caught.exception.code, row["expected"])

    def test_dataset_labels_preserve_every_raw_outcome_and_loss_mask(self):
        from nca.ik.data import label_configuration
        labels = []
        for row in self.fixture["synthetic_two_link"]["cases"]:
            with self.subTest(case=row["id"]):
                label = label_configuration(self.synthetic, row["target"], row["current"],
                    posture=row["posture"], solver_config=SolverConfig(
                        max_seconds=1., max_iterations=row.get("max_iterations", 64)))
                self.assertEqual(label.teacher.status, row["numeric_status"])
                self.assertEqual(label.analytic.status, row["analytic_status"])
                self.assertEqual(label.pose_loss, row["pose_loss"])
                self.assertTrue(label.status_loss)
                if row["pose_loss"]:
                    self.assertTrue(label.verified)
                    self.assertLessEqual(label.endpoint_error_cells, 1e-6)
                    self.assertLess(math.dist(label.independent_endpoint, row["target"]), 1e-6)
                labels.append(label)
        self.assertEqual(len(labels), 7)
        self.assertEqual(sum(label.pose_loss for label in labels), 5)
        self.assertNotEqual(labels[0].branch_id, labels[1].branch_id)
        # The analytic reachable answer must not replace the exhausted DLS call.
        self.assertEqual(labels[-1].teacher.status, "iteration_budget_exhausted")
        self.assertFalse(labels[-1].pose_loss)

    def test_dataset_branch_identity_uses_vectors_not_runner_ankle_sign(self):
        from nca.ik.data import branch_identity, independent_two_link_point
        chain = chain_from_profile(self.profile)
        branches = {branch_identity(chain, (.1, ankle)) for ankle in (-.27, -.1, 0., .1, .27)}
        self.assertEqual(len(branches), 1)
        synthetic_cases = self.fixture["synthetic_two_link"]["cases"]
        positive = branch_identity(self.synthetic, synthetic_cases[0]["expected_q"])
        negative = branch_identity(self.synthetic, synthetic_cases[1]["expected_q"])
        singular = branch_identity(self.synthetic, (0., 0.))
        self.assertEqual(branches, {negative})
        self.assertEqual(len({positive, negative, singular}), 3)
        for row in self.fixture["runner"]["poses"]:
            self.assertLess(math.dist(independent_two_link_point(chain, row["q"]), row["point"]), 1e-12)

    def test_dataset_label_does_not_invent_infeasibility_from_time_exhaustion(self):
        from nca.ik.data import label_configuration
        label = label_configuration(self.synthetic, (1., 1.), (0., .1),
                                    solver_config=SolverConfig(max_seconds=0.))
        self.assertEqual(label.teacher.status, "time_budget_exhausted")
        self.assertEqual(label.analytic.status, "solved")
        self.assertFalse(label.pose_loss)
        self.assertTrue(label.status_loss)


if __name__ == "__main__":
    unittest.main()
