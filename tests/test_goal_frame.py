"""Declared coordinate and evidence contracts; fixtures are not gameplay scores."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

from nca.contracts import ContractError, WireAction, sha256_json
from nca.train.goal_frame_config import (FOLLOWUP_SHA256, read_followup_config,
                                        summarize_prerequisite, validate_followup_config,
                                        verify_original_boundary)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/first-skill/jolt-direction-frame-v1.json"
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from nca.controllers.baseline import SkillGoal
    from nca.controllers.goal_frame import (SIGNED_FEATURES, GoalFrameController, GoalFramePolicy,
        initial_direction, load_goal_frame_policy, restore_request, transform_features,
        transform_labels, transform_sequences)
    from nca.controllers.learned import load_policy, policy_state_sha256
    from nca.data import ActionSource, CaptureEvent, EpisodeRecord, TickRecord, policy_view
    from nca.train.artifacts import write_checkpoint
    from nca.train.goal_frame import audit_episode_boundaries, verify_checkpoint_file, verify_published_files
    from nca.train.recurrent import train_clone
    from nca.train.screen import ScreenBudgetExceeded
    from test_learning import view
    from test_morphology import fixture


def setUpModule():
    if TORCH_AVAILABLE:
        torch.set_num_threads(1)


def complete_outcomes(config):
    """Exact full synthetic schedule for gate logic, never a measured result."""
    rows = []
    for index, case in enumerate(config["evaluation_families"]):
        policies = [("baseline", "baseline", index < 2), ("idle", "idle", False)]
        for family in config["candidates_in_priority_order"]:
            for seed in config["training_seeds"]:
                for phase in ("initial", "final"):
                    policies.append((f"{family}-seed{seed}-{phase}", family, phase == "final" and index < 6))
        for policy_id, kind, success in policies:
            rows.append({"split": "evaluation", "scenario_family": case["id"], "host_seed": case["host_seed"],
                         "policy_id": policy_id, "kind": kind, "source_boundary_regressions": 0, "error": None,
                         "metrics": {"success": success, "capped_completion_seconds": 2.0 if success else 4.0,
                             "ordinary_trim_reversals_per_second": 0.0, "mean_absolute_angle_radians": .1,
                             "peak_toward_speed_cells_per_second": 10.0, "purposeful_launch": kind != "idle",
                             "inference_p95_ms": 1.0, "return": -.5 if kind == "idle" else .1,
                             "integrity_failure": False, "failed_admissions": 0,
                             "administratively_capped": False}})
    return rows


class GoalFrameDeclarationTests(unittest.TestCase):
    def setUp(self):
        self.config = read_followup_config(CONFIG_PATH)

    def test_new_geometry_and_all_original_thresholds_are_bound_before_work(self):
        self.assertEqual(sha256_json(self.config), FOLLOWUP_SHA256)
        verify_original_boundary(self.config, ROOT)
        self.assertEqual(self.config["training_seeds"], [7103, 7207, 7301])
        self.assertEqual(self.config["budget"]["planned_worst_case_decisions"], 5440)
        self.assertEqual(self.config["prerequisite"]["minimum_mean_success"], .75)
        for key in ("training_seeds", "budget", "prerequisite"):
            changed = deepcopy(self.config)
            changed[key] = []
            with self.subTest(key=key), self.assertRaises(ContractError):
                validate_followup_config(changed)

    def test_complete_families_use_declared_priority_and_all_seed_resamples(self):
        result = summarize_prerequisite(complete_outcomes(self.config), self.config, run_completed=True)
        self.assertTrue(result["mechanical_utility_prerequisite_passed"])
        self.assertEqual(result["selected_family"], "direction_frame_bc")
        self.assertEqual(set(result["candidates"]), set(self.config["candidates_in_priority_order"]))
        for candidate in result["candidates"].values():
            self.assertEqual(candidate["mean_success"], .75)
            self.assertEqual(candidate["gain_over_initial"]["resamples"], 27)
            self.assertEqual(len(candidate["per_seed"]), 3)
        self.assertFalse(result["original_entry_result_changed"])
        self.assertFalse(result["learned_morphology_transfer_measured"])

    def test_family_mean_does_not_hide_one_failed_seed(self):
        rows = complete_outcomes(self.config)
        for row in rows:
            if row["policy_id"].startswith("direction_frame_bc") and row["policy_id"].endswith("final"):
                index = next(i for i, case in enumerate(self.config["evaluation_families"])
                             if case["id"] == row["scenario_family"])
                row["metrics"]["success"] = index < 3 if "seed7103" in row["policy_id"] else True
        result = summarize_prerequisite(rows, self.config, run_completed=True)
        failed = result["candidates"]["direction_frame_bc"]
        self.assertGreater(failed["mean_success"], .75)
        self.assertEqual(failed["minimum_seed_success"], .375)
        self.assertFalse(failed["passed"])
        self.assertIn("minimum_every_seed_success", failed["failed"])
        self.assertEqual(result["selected_family"], "world_frame_bc")

    def test_initial_gain_and_every_scenario_runtime_are_mandatory(self):
        for mutation in ("no_gain", "latency", "admission", "censored", "integrity"):
            rows = complete_outcomes(self.config)
            for row in rows:
                if row["kind"] != "direction_frame_bc":
                    continue
                if mutation == "no_gain" and row["policy_id"].endswith("initial"):
                    row["metrics"]["success"] = True
                elif row["policy_id"] == "direction_frame_bc-seed7103-final":
                    if mutation == "latency":
                        row["metrics"]["inference_p95_ms"] = 5.01
                    elif mutation == "admission":
                        row["metrics"]["failed_admissions"] = 1
                    elif mutation == "censored":
                        row["metrics"]["administratively_capped"] = True
                    elif mutation == "integrity":
                        row["metrics"]["integrity_failure"] = True
            result = summarize_prerequisite(rows, self.config, run_completed=True)
            with self.subTest(mutation=mutation):
                self.assertFalse(result["candidates"]["direction_frame_bc"]["passed"])
                self.assertEqual(result["selected_family"], "world_frame_bc")

    def test_incomplete_or_malformed_evidence_cannot_admit_any_family(self):
        for mutation in ("missing", "duplicate", "nan", "source", "boolean_count", "kind", "seed", "error", "negative"):
            rows = complete_outcomes(self.config)
            if mutation == "missing":
                rows.pop()
            elif mutation == "duplicate":
                rows[-1] = deepcopy(rows[0])
            elif mutation == "nan":
                rows[-1]["metrics"]["mean_absolute_angle_radians"] = float("nan")
            elif mutation == "source":
                rows[-1]["source_boundary_regressions"] = 1
            elif mutation == "boolean_count":
                rows[-1]["source_boundary_regressions"] = False
            elif mutation == "kind":
                rows[-1]["kind"] = "bc"
            elif mutation == "seed":
                rows[-1]["host_seed"] += 1
            elif mutation == "negative":
                rows[-1]["metrics"]["inference_p95_ms"] = -1.
            else:
                rows[-1]["error"] = {"type": "HostError"}
            result = summarize_prerequisite(rows, self.config, run_completed=True)
            with self.subTest(mutation=mutation):
                self.assertFalse(result["mechanical_utility_prerequisite_passed"])
                self.assertTrue(result["errors"])
        partial = summarize_prerequisite(complete_outcomes(self.config), self.config, run_completed=False)
        self.assertFalse(partial["mechanical_utility_prerequisite_passed"])


@unittest.skipUnless(TORCH_AVAILABLE, "optional CPU learning extra is not installed")
class GoalFrameControllerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.goal = SkillGoal("hop", "directed_hop", (-48., 0.), timeout_seconds=4)

    def test_coordinate_label_and_action_maps_are_involutions(self):
        features = [float(index) / 26 for index in range(26)]
        labels = [.25, -.75, .8, 1., 0., 1.]
        action = WireAction(move=(.25, .2), aim=(-.6, -.8), trim=-.75, effort=.8,
                            primary=True, auxiliary=True)
        for direction in (-1, 1):
            first = transform_features(features, direction)
            self.assertEqual(transform_features(first, direction), tuple(features))
            self.assertTrue(all(first[index] == value for index, value in enumerate(features)
                                if index not in SIGNED_FEATURES))
            self.assertEqual(transform_labels(transform_labels(labels, direction), direction), tuple(labels))
            self.assertEqual(restore_request(restore_request(action, direction), direction), action)
            self.assertEqual(restore_request(action, direction).aim, action.aim)
        self.assertEqual(initial_direction(0.), 1)
        for invalid in (0, True, -1.0):
            with self.assertRaises(ContractError):
                transform_features(features, invalid)
        with self.assertRaises(ContractError):
            transform_features(features[:-1], 1)
        with self.assertRaises(ContractError):
            transform_labels([float("nan")] + labels[1:], 1)

    def test_reflected_input_algebra_preserves_canonical_recurrence_and_inverse_action(self):
        policy = GoalFramePolicy("direction_frame_bc")
        right, left = GoalFrameController(policy), GoalFrameController(policy)
        for tick in (0, 6, 12):
            a = right.step(view(tick, velocity=[3., 4.], rotation=.2, angular_velocity=1.),
                           replace(self.goal, target=(48., 0.)))
            b = left.step(view(tick, velocity=[-3., 4.], rotation=-.2, angular_velocity=-1.), self.goal)
            self.assertEqual(a.features, b.features)
            self.assertEqual(a.hidden_before, b.hidden_before)
            torch.testing.assert_close(right.hidden, left.hidden, rtol=0, atol=0)
            self.assertEqual(a.action.move[0], -b.action.move[0])
            self.assertEqual(a.action.trim, -b.action.trim)
            self.assertEqual(a.action.aim[0], -b.action.aim[0])
            self.assertEqual(a.action.aim[1], b.action.aim[1])

    def test_direction_is_fixed_after_overshoot_and_history_is_actual_world_request(self):
        controller = GoalFrameController(GoalFramePolicy("direction_frame_bc"))
        first = controller.step(view(), self.goal)
        self.assertEqual(controller.direction, -1)
        second = controller.step(view(6, position=[-60., 0.]), self.goal)
        self.assertEqual(controller.direction, -1)
        self.assertLess(second.features[0], 0)
        self.assertEqual(second.features[20], -first.action.move[0])
        self.assertEqual(second.features[21], -first.action.trim)
        self.assertEqual(controller.previous, second.action)
        fresh = controller.step(view(12, position=[-60., 0.]), replace(self.goal, id="fresh", target=(0., 0.)))
        self.assertEqual(controller.direction, 1)
        self.assertEqual(fresh.hidden_before, (0.,) * 32)

    def test_original_cancel_takeover_lifetime_and_goal_fences_remain_active(self):
        for boundary in ("cancel", "takeover", "goal_change", "repeat_tick"):
            controller = GoalFrameController(GoalFramePolicy("direction_frame_bc"))
            controller.step(view(), self.goal)
            changed, goal = view(6), self.goal
            if boundary == "cancel":
                controller.cancel()
            elif boundary == "takeover":
                changed = view(6, control_generation=1)
            elif boundary == "goal_change":
                goal = replace(self.goal, target=(20., 0.))
            else:
                changed = view()
            with self.subTest(boundary=boundary), self.assertRaises(ValueError):
                controller.step(changed, goal)
        controller = GoalFrameController(GoalFramePolicy("direction_frame_bc"))
        controller.step(view(), self.goal)
        replaced = controller.step(view(run_id="replacement"), replace(self.goal, target=(48., 0.)))
        self.assertEqual(controller.direction, 1)
        self.assertEqual(replaced.hidden_before, (0.,) * 32)
        with self.assertRaises(ContractError):
            controller.step(view(6, run_id="replacement"), self.goal, deterministic=False)

    def test_same_complete_training_sequences_receive_one_fixed_direction(self):
        rows = [[-.5] + [0.] * 25, [.25] + [.1] * 25]
        labels = [[-.8, -.2, .7, 1., 0., 0.], [.4, .1, .3, 0., 1., 0.]]
        original = deepcopy((rows, labels))
        canonical = transform_sequences([(rows, labels)], "direction_frame_bc")
        self.assertEqual(len(canonical[0][0]), 2)
        self.assertEqual(canonical[0][0][1][0], -.25)
        self.assertEqual(canonical[0][1][0][0], .8)
        self.assertEqual((rows, labels), original)
        world = transform_sequences([(rows, labels)], "world_frame_bc")
        self.assertEqual(world[0][0], [tuple(row) for row in rows])

    def test_checkpoint_contract_prevents_relabeling_and_preserves_sequential_outputs(self):
        policies = []
        for family in ("direction_frame_bc", "world_frame_bc"):
            torch.manual_seed(7103)
            policies.append(GoalFramePolicy(family))
        self.assertTrue(all(torch.equal(value, policies[1].state_dict()[name])
                            for name, value in policies[0].state_dict().items()))
        self.assertNotEqual(policy_state_sha256(policies[0]), policy_state_sha256(policies[1]))
        with tempfile.TemporaryDirectory() as directory:
            for policy in policies:
                metadata = {"config_sha256": FOLLOWUP_SHA256, "family": policy.family, "feature_frame": policy.frame}
                path = Path(directory) / f"{policy.family}.pt"
                write_checkpoint(path, policy, metadata)
                restored, _ = load_goal_frame_policy(path, expected_config_sha256=FOLLOWUP_SHA256)
                left, right = GoalFrameController(policy), GoalFrameController(restored)
                for tick in (0, 6, 12):
                    self.assertEqual(left.step(view(tick), self.goal), right.step(view(tick), self.goal))
                with self.assertRaises(ValueError):
                    load_policy(path, expected_config_sha256=FOLLOWUP_SHA256)
                with self.assertRaises(ContractError):
                    load_goal_frame_policy(path, expected_config_sha256="0" * 64)
            wrong = Path(directory) / "wrong-frame.pt"
            write_checkpoint(wrong, policies[0], dict(metadata, family=policies[0].family))
            with self.assertRaises(ContractError):
                load_goal_frame_policy(wrong, expected_config_sha256=FOLLOWUP_SHA256)

    def test_budget_abort_precedes_next_update_and_records_completed_updates(self):
        policy = GoalFramePolicy("direction_frame_bc")
        rows, labels = [[0.] * 26] * 3, [[.4, -.2, .7, 1., 0., 0.]] * 3
        before, updates = policy_state_sha256(policy), []
        def stop():
            raise ScreenBudgetExceeded("fixed test ceiling")
        with self.assertRaises(ScreenBudgetExceeded):
            train_clone(policy, [(rows, labels)], seed=1, epochs=1, unroll=1, check_budget=stop)
        self.assertEqual(policy_state_sha256(policy), before)
        def allow_one():
            if updates:
                stop()
        with self.assertRaises(ScreenBudgetExceeded):
            train_clone(policy, [(rows, labels)], seed=1, epochs=1, unroll=1,
                        check_budget=allow_one, on_update=updates.append)
        self.assertEqual(updates, [1])
        self.assertNotEqual(policy_state_sha256(policy), before)

    def test_checkpoint_file_replacement_and_untracked_source_are_refused(self):
        policy = GoalFramePolicy("world_frame_bc")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / "checkpoints" / "candidate.pt"
            receipt = write_checkpoint(path, policy, {"config_sha256": FOLLOWUP_SHA256})
            self.assertEqual(verify_checkpoint_file(output, receipt), path)
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(data)
            with self.assertRaises(ContractError):
                verify_checkpoint_file(output, receipt)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            source = root / "source.py"
            source.write_bytes(b"value = 1\n")
            subprocess.run(["git", "add", "source.py"], cwd=root, check=True)
            subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                            "commit", "-qm", "source fixture"], cwd=root, check=True)
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            verify_published_files(root, revision, {"source.py": digest})
            source.write_bytes(b"value = 2\n")
            with self.assertRaises(ContractError):
                verify_published_files(root, revision, {"source.py": hashlib.sha256(source.read_bytes()).hexdigest()})
            (root / "untracked.py").write_bytes(b"value = 3\n")
            with self.assertRaises(ContractError):
                verify_published_files(root, revision, {"untracked.py": "0" * 64})


def boundary_episode():
    adapter, before = fixture()
    before["joints"][0]["target"] = .5
    source = ActionSource("teacher", "scripted", "teacher", "a" * 64)
    action = WireAction(move=(.35, 0), effort=.6).to_dict()
    packet = {"version": 2, "kind": "physical_wire_action", "run_id": before["run_id"],
              "epoch": 1, "actor_id": 0, "sequence": 0, "observation_id": before["observation_id"],
              "observed_tick": 0, "lease_ticks": 12, "action": action}
    ticks = []
    for tick in range(6):
        _, observation = fixture(tick=tick + 1)
        observation["joints"][0]["target"] = .5
        ticks.append(TickRecord(tick, tick + 1, tick + 1, 10 + tick, 11 + tick,
            (packet,) if tick == 0 else (), ({"accepted": True, "reason": ""},) if tick == 0 else (),
            packet, source.id, action, {"action_available": True, "terminal_revocation": False, "action": action},
            {"available": False, "reason": "fixture"}, {}, observation, None))
    after = ticks[-1].observation
    step = CaptureEvent("step", 0, 0, 0, 6, 0, 20, source, before, policy_view(before),
                        requested_packet=packet, ticks=tuple(ticks), next_observation=after, source_observation=before)
    _, cancelled = fixture(tick=6, control=1)
    cancel = CaptureEvent("cancel", 1, 1, 6, 6, 20, 21, source, after, policy_view(after),
                          next_observation=cancelled, reason="fixture_stop")
    return EpisodeRecord("fixture-episode", {"lane": "act004"}, "synthetic", "a" * 64, {}, adapter.profile,
                         before, (step, cancel), "completed", "fixture", "fixture", "training")


@unittest.skipUnless(TORCH_AVAILABLE, "optional CPU learning extra is not installed")
class GoalFrameReceiptTests(unittest.TestCase):
    def test_original_admitted_interval_and_same_tick_motor_revocation_pass(self):
        episode = boundary_episode()
        result = audit_episode_boundaries(episode, episode.profile_sha256)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["completed_ticks"], 6)
        self.assertEqual(result["cancellations"], 1)

    def test_consumed_packet_replacement_and_false_cancel_fail(self):
        episode = boundary_episode()
        step, cancel = episode.events
        consumed = dict(step.ticks[2].consumed_request, observation_id="another-original-view")
        changed_tick = replace(step.ticks[2], consumed_request=consumed)
        changed_step = replace(step, ticks=(*step.ticks[:2], changed_tick, *step.ticks[3:]))
        result = audit_episode_boundaries(replace(episode, events=(changed_step, cancel)), episode.profile_sha256)
        self.assertFalse(result["passed"])
        self.assertEqual(result["source_boundary_regressions"], 1)
        _, changed_observation = fixture(tick=6, control=1)
        changed_observation["parts"][0]["velocity"][0] = .1
        bad_cancel = replace(cancel, next_observation=changed_observation)
        result = audit_episode_boundaries(replace(episode, events=(step, bad_cancel)), episode.profile_sha256)
        self.assertFalse(result["passed"])
        missing_cancel = replace(episode, events=(step,))
        self.assertFalse(audit_episode_boundaries(missing_cancel, episode.profile_sha256)["passed"])


if __name__ == "__main__":
    unittest.main()
