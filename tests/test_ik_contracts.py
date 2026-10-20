"""Nominal protocol fixtures only: no CUDA, host control or physical evidence."""

import contextlib
import io
import json
import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from nca.builtins import builtin_registry
from nca.cli import main
from nca.contracts import ContractError, LifecycleStamp, sha256_json
from nca.ik import (IKAuthority, IKObservation, IKResult, IKTarget, InfeasibilityProof,
                    JointReference, RootPose, point_to_root, target_from_observation)
from nca.ik.profiles import nominal_outer_reach, nominal_pose, runner_right_sole_profile


POLICY = sha256_json({"kind": "nominal-fixture", "version": 1})


def observation(profile, **changes):
    obs = IKObservation(
        id="issued-10", tick=10, lifecycle=LifecycleStamp("ik-fixture", "runner-0", 1),
        profile_sha256=profile.sha256, source_sha256=profile.source_sha256,
        topology_sha256=profile.topology_sha256, root_pose=RootPose((10., 20.), math.pi / 2),
        part_present=(True,) * 5 + (False,) * 3,
        joint_present=(True,) * 4 + (False,) * 3,
        joint_measured=(False, False, True, True, False, False, False),
        joint_active=(False, False, True, True, False, False, False),
        joint_positions=(None, None, 0., 0., None, None, None))
    return replace(obs, **changes)


def target(profile, obs, **changes):
    values = dict(id="target-0", source_id="fixture-controller", goal_id="reach-0",
                  sequence=0, policy_sha256=POLICY, grant_id="grant-1", grant_generation=1,
                  point=(4.6, 9.), goal_deadline_tick=100)
    values.update(changes)
    return target_from_observation(profile, obs, **values)


def authority(obs, **changes):
    value = IKAuthority(obs.tick, obs.lifecycle, "fixture-controller", "reach-0",
                        POLICY, "grant-1", 1)
    return replace(value, **changes)


def candidate(profile, goal, positions=(0., 0.), status="solved", kind="joint_position", obs=None):
    coordinates = dict(zip(profile.solver_joint_ids, positions))
    if kind == "joint_delta":
        current = obs.configuration(profile)
        coordinates = {key: current[key] + value for key, value in coordinates.items()}
    pose = nominal_pose(profile, coordinates)
    return IKResult(goal.sha256, "hand-fixture", status, "independent nominal fixture",
                    reference=JointReference(profile.solver_joint_ids, positions, kind),
                    achieved_point_root=pose.point,
                    residual_cells=math.dist(pose.point, goal.point_root),
                    orientation_radians=pose.orientation_radians)


class IKContractTests(unittest.TestCase):
    def setUp(self):
        self.profile = runner_right_sole_profile()
        self.obs = observation(self.profile)
        self.target = target(self.profile, self.obs)
        self.authority = authority(self.obs)

    def assertCode(self, expected, function, *args, **kwargs):
        with self.assertRaises(ContractError) as caught:
            function(*args, **kwargs)
        self.assertEqual(expected, caught.exception.code)

    def validate_target(self, value=None, obs=None, auth=None):
        (value or self.target).validate_against(self.profile, obs or self.obs, auth or self.authority)

    def validate_result(self, value, goal=None, obs=None, auth=None):
        value.validate_against(self.profile, goal or self.target, obs or self.obs, auth or self.authority)

    def test_roundtrip_schema_and_immutable_records(self):
        result = candidate(self.profile, self.target)
        for value in (self.obs, self.target, self.authority, result):
            with self.subTest(record=type(value).__name__):
                self.assertEqual(type(value).from_json(value.to_json()), value)
                self.assertEqual(type(value).from_dict(value.to_dict()).sha256, value.sha256)
                schema = type(value).json_schema()
                self.assertFalse(schema["$defs"][type(value).__name__]["additionalProperties"])
        self.validate_result(result)
        with self.assertRaises(FrozenInstanceError):
            self.target.sequence = 1

    def test_registered_profile_has_no_claimed_physical_or_cuda_capability(self):
        registry = builtin_registry()
        loaded = registry.create("profile", "runner-right-sole", require=("point-ik-2d",))
        self.assertEqual(loaded, self.profile)
        for missing in ("cuda", "joint-targets", "physical-qualified"):
            self.assertCode("missing_capability", registry.create, "profile", "runner-right-sole", require=(missing,))
        script = ("import sys,json;from nca.builtins import builtin_registry;"
                  "p=builtin_registry().create('profile','runner-right-sole');"
                  "print(json.dumps({'id':p.id,'optional':sorted(set(sys.modules)&{'torch','numpy'})}))")
        data = json.loads(subprocess.check_output([sys.executable, "-c", script], text=True))
        self.assertEqual(data, {"id": "runner-right-sole", "optional": []})

    def test_presence_measurement_and_participation_are_not_interchangeable(self):
        self.assertEqual(self.obs.configuration(self.profile), {"hip_1": 0., "ankle_1": 0.})
        self.assertCode("invalid_mask", replace, self.obs,
                        joint_positions=(0., None, 0., 0., None, None, None))
        missing = replace(self.obs, joint_measured=(False,) * 7,
                          joint_positions=(None,) * 7)
        self.assertCode("missing_required_measurement", missing.validate_profile, self.profile)
        self.assertCode("missing_required_measurement", replace(self.obs, root_pose=None).validate_profile, self.profile)
        self.assertCode("topology_mismatch", replace(self.obs,
                        part_present=(True, False, True, True, True, False, False, False)).validate_profile, self.profile)
        self.assertCode("activity_mismatch", replace(self.obs,
                        joint_active=(True, False, True, True, False, False, False)).validate_profile, self.profile)
        self.assertCode("invalid_mask", replace, self.obs,
                        joint_present=(True, True, False, True, False, False, False))
        self.assertCode("shape_mismatch", replace, self.obs, part_present=(True,) * 5)

    def test_measured_play_is_retained_without_claiming_a_legal_reference(self):
        # Actual compliant motion is not silently clamped to a desired interval.
        moved = replace(self.obs, joint_positions=(None, None, .78, .31, None, None, None))
        self.assertEqual(moved.configuration(self.profile), {"hip_1": .78, "ankle_1": .31})
        for value in (math.pi, math.pi + .01):
            self.assertCode("out_of_range", replace, moved,
                            joint_positions=(None, None, value, .31, None, None, None))
        wrapped = replace(moved, joint_positions=(None, None, -math.pi, .31, None, None, None))
        self.assertEqual(wrapped.configuration(self.profile)["hip_1"], -math.pi)

    def test_world_conversion_uses_issued_root_and_down_y_rotation(self):
        # R(pi/2)*(4.6,9)+(10,20) = (1,24.6).
        goal = target(self.profile, self.obs, point=(1., 24.6), input_frame="world")
        self.assertAlmostEqual(goal.point_root[0], 4.6)
        self.assertAlmostEqual(goal.point_root[1], 9.)
        self.validate_target(goal)
        refreshed = replace(self.obs, root_pose=RootPose((11., 20.), math.pi / 2))
        self.assertCode("observation_mismatch", self.validate_target, goal, refreshed)
        self.assertCode("frame_mismatch", self.validate_target, replace(goal, point_root=(4.6, 8.)))
        self.assertCode("unsupported_frame", point_to_root, (0, 0), "effector", self.obs)
        self.assertCode("missing_required_measurement", point_to_root, (0, 0), "world",
                        replace(self.obs, root_pose=None))

    def test_exclusive_age_lease_and_goal_deadline_do_not_refresh(self):
        self.validate_target(auth=replace(self.authority, current_tick=21))
        self.assertCode("stale_observation", self.validate_target, auth=replace(self.authority, current_tick=22))
        longer = replace(self.target, lease_ticks=18)
        self.assertEqual(longer.expires_at(self.profile), 22)
        self.assertCode("stale_observation", self.validate_target, longer, auth=replace(self.authority, current_tick=22))
        shorter = replace(self.target, lease_ticks=6)
        self.validate_target(shorter, auth=replace(self.authority, current_tick=15))
        self.assertCode("expired_target", self.validate_target, shorter, auth=replace(self.authority, current_tick=16))
        deadline = replace(self.target, goal_deadline_tick=13)
        self.assertCode("expired_target", self.validate_target, deadline, auth=replace(self.authority, current_tick=13))
        self.assertCode("future_observation", self.validate_target, auth=replace(self.authority, current_tick=9))
        self.assertCode("observation_mismatch", self.validate_target, replace(self.target, observed_tick=11))
        for value in (0, 19, True):
            with self.assertRaises(ContractError):
                replace(self.target, lease_ticks=value)

    def test_all_lifecycle_fields_fence_old_results(self):
        changes = dict(run_id="other-run", actor_id="runner-1", epoch=2, incarnation=1,
                       control_generation=1, morphology_version=2, mount_generation=1)
        for field, value in changes.items():
            with self.subTest(field=field):
                life = replace(self.authority.lifecycle, **{field: value})
                self.assertCode("lifecycle_mismatch", self.validate_target,
                                auth=replace(self.authority, lifecycle=life))
        self.assertCode("observation_mismatch", self.validate_target, replace(self.target, observation_id="other"))

    def test_takeover_policy_grant_goal_cancel_and_replay_are_separate(self):
        for field, value in (("source_id", "different-source"), ("goal_id", "different-goal"),
                             ("grant_id", "different-grant"), ("grant_generation", 2)):
            self.assertCode("authority_mismatch", self.validate_target,
                            auth=replace(self.authority, **{field: value}))
        self.assertCode("policy_mismatch", self.validate_target,
                        auth=replace(self.authority, policy_sha256="f" * 64))
        self.assertCode("cancelled", self.validate_target, auth=replace(self.authority, revoked=True))
        self.assertCode("already_consumed", self.validate_target,
                        auth=replace(self.authority, last_consumed_sequence=0))
        self.validate_target(replace(self.target, sequence=1),
                             auth=replace(self.authority, last_consumed_sequence=0))

    def test_source_profile_topology_and_joint_order_are_bound(self):
        for field, code in (("profile_sha256", "profile_mismatch"), ("source_sha256", "source_mismatch"),
                            ("topology_sha256", "topology_mismatch")):
            self.assertCode(code, self.validate_target, replace(self.target, **{field: "a" * 64}))
            self.assertCode(code, replace(self.obs, **{field: "a" * 64}).validate_profile, self.profile)
        self.assertCode("unsupported_profile", self.validate_target,
                        replace(self.target, joint_ids=tuple(reversed(self.target.joint_ids))))
        self.assertCode("unsupported_profile", self.validate_target, replace(self.target, effector_id="left_sole"))
        self.assertCode("out_of_range", self.validate_target, replace(self.target, posture_preference=(.73, 0)))
        self.validate_target(replace(self.target, posture_preference=(.6, -.2)))

    def test_delta_bound_mode_and_absolute_geometric_goal_are_distinct(self):
        goal = target(self.profile, self.obs, point=nominal_pose(self.profile, {"hip_1": .25, "ankle_1": 0.}).point,
                      output_kind="joint_delta")
        result = candidate(self.profile, goal, (.25, 0.), kind="joint_delta", obs=self.obs)
        self.validate_result(result, goal)
        wrong_mode = replace(result, reference=replace(result.reference, kind="joint_position"))
        self.assertCode("reference_mismatch", self.validate_result, wrong_mode, goal)
        too_large = candidate(self.profile, goal, (.251, 0), status="best_effort", kind="joint_delta", obs=self.obs)
        self.assertCode("out_of_range", self.validate_result, too_large, goal)
        # Absolute nominal solutions can be farther away; the future host owns slew.
        full_point = nominal_pose(self.profile, {"hip_1": .6, "ankle_1": 0.}).point
        absolute = target(self.profile, self.obs, point=full_point)
        self.validate_result(candidate(self.profile, absolute, (.6, 0)), absolute)

    def test_results_match_references_residual_tolerance_and_inner_limits(self):
        result = candidate(self.profile, self.target)
        self.assertCode("result_mismatch", self.validate_result, replace(result, achieved_point_root=(0, 0)))
        self.assertCode("result_mismatch", self.validate_result, replace(result, orientation_radians=.1))
        self.assertCode("result_mismatch", self.validate_result, replace(result, residual_cells=.1))
        self.assertCode("status_mismatch", self.validate_result, replace(result, status="best_effort"))
        best = candidate(self.profile, self.target, (.1, 0), "best_effort")
        self.validate_result(best)
        self.assertCode("status_mismatch", self.validate_result, replace(best, status="solved"))
        outside = candidate(self.profile, self.target, (.74, 0), "best_effort")
        self.assertCode("out_of_range", self.validate_result, outside)
        self.assertCode("target_mismatch", self.validate_result, replace(result, target_sha256="a" * 64))
        self.assertCode("stale_observation", self.validate_result, result, auth=replace(self.authority, current_tick=22))

    def test_infeasibility_requires_independently_checked_geometric_proof(self):
        center, radius = nominal_outer_reach(self.profile)
        goal = target(self.profile, self.obs, point=(center[0] + radius + 1, center[1]))
        proof = InfeasibilityProof(center, radius, radius + 1)
        result = IKResult(goal.sha256, "radial-bound", "geometrically_infeasible",
                          "outside unrestricted reach radius", proof=proof)
        self.validate_result(result, goal)
        forged = replace(result, proof=replace(proof, outer_radius_cells=radius - 1))
        self.assertCode("invalid_proof", self.validate_result, forged, goal)
        reachable = replace(result, target_sha256=self.target.sha256)
        self.assertCode("invalid_proof", self.validate_result, reachable)
        self.assertCode("invalid_proof", replace, result, status="iteration_budget_exhausted")
        self.assertCode("invalid_proof", replace, result, proof=None)

    def test_failed_or_invalid_results_cannot_carry_actions(self):
        result = candidate(self.profile, self.target)
        for status in ("iteration_budget_exhausted", "invalid_input", "unsupported_profile",
                       "missing_required_measurement", "nonfinite_result"):
            with self.subTest(status=status):
                failed = IKResult(self.target.sha256, "failed-fixture", status, "explicit refusal")
                self.validate_result(failed)
                self.assertCode("rejected_payload", replace, failed, reference=result.reference)
        with self.assertRaises(ContractError):
            replace(result, reference=replace(result.reference, values=(float("nan"), 0)))

    def test_unknown_orientation_objective_and_wrong_units_are_refused(self):
        value = self.target.to_dict()
        value["orientation_target"] = 0.
        self.assertCode("unknown_field", IKTarget.from_dict, value)
        value = candidate(self.profile, self.target).to_dict()
        value["reference"]["unit"] = "radians_per_second"
        with self.assertRaises(ContractError):
            IKResult.from_dict(value)
        for value in (True, float("inf")):
            with self.assertRaises(ContractError):
                replace(self.target, point_root=(value, 9))

    def test_cli_structural_validation_is_not_contextual_admission(self):
        records = (("ik-profile", self.profile), ("ik-observation", self.obs),
                   ("ik-target", self.target), ("ik-result", candidate(self.profile, self.target)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            for kind, record in records:
                path.write_text(record.to_json(), encoding="utf-8")
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main(["validate", kind, str(path)]), 0)
                self.assertEqual(json.loads(output.getvalue())["sha256"], record.sha256)
            path.write_text('{"schema_version":2}', encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["validate", "ik-target", str(path)]), 2)


if __name__ == "__main__":
    unittest.main()
