"""Morphology contracts and closed-gate tests; fixtures are not transfer scores."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from nca.contracts import ContractError, sha256_json, strict_json_loads
from nca.controllers.morphology import (
    EDGE_NAMES, ROOT_NAMES, MorphologyAdapter, MorphologyGoal, SemanticAction,
    StateFence, information_reach, map_action,
)
from nca.controllers.morphology.admission import (
    COMPARISON_SHA256, comparison_plan, entry_decision, prepare_comparison,
)

ROOT = Path(__file__).resolve().parents[1]


def fixture(model="jev", *, tick=0, epoch=1, control=0):
    row = strict_json_loads((ROOT / "configs/morphology/profiles-v1.json").read_bytes())["profiles"][model]
    adapter = MorphologyAdapter(row["definition"], expected_sha256=row["profile_sha256"])
    view = deepcopy(row["observation"])
    view.update(run_id="morphology-fixture", tick=tick, epoch=epoch, control_generation=control)
    view["observation_id"] = f"{view['run_id']}:{epoch}:{view['actor_id']}:{control}:{tick}"
    return adapter, view


def goal(view, name="test-waypoint"):
    return MorphologyGoal(name, (view["parts"][0]["position"][0] + 40, view["parts"][0]["position"][1]))


class MorphologyContractTests(unittest.TestCase):
    def test_four_actual_profile_fixtures_and_padding(self):
        expected = {"jev": (3, 2), "astra": (3, 2), "monowheel": (2, 1), "ripper": (5, 4)}
        for model, (parts, edges) in expected.items():
            adapter, view = fixture(model)
            frame = adapter.frame(view, goal(view))
            with self.subTest(model=model):
                self.assertEqual(sum(frame.present), parts)
                self.assertEqual(sum(edge.present for edge in frame.edges), edges)
                self.assertEqual(len(frame.padded()), 855)
                self.assertTrue(all(not any(row) for row in frame.nodes[parts:]))
                self.assertEqual(information_reach(frame, 2), tuple(range(parts)))

    def test_profile_and_source_are_owned_copies(self):
        row = strict_json_loads((ROOT / "configs/morphology/profiles-v1.json").read_bytes())["profiles"]["jev"]
        adapter = MorphologyAdapter(row["definition"], expected_sha256=row["profile_sha256"])
        row["definition"]["parts"][0]["mass"] = 999
        self.assertAlmostEqual(adapter.parts[0]["mass"], 1.3)
        with self.assertRaises(TypeError):
            adapter.profile["parts"][0]["mass"] = 999
        with self.assertRaises(TypeError):
            adapter.capability["impulse"] = "secondary"
        _, view = fixture()
        frame = adapter.frame(view, goal(view))
        old = frame.nodes[0]
        view["parts"][0]["velocity"][0] = 100
        self.assertEqual(frame.nodes[0], old)

    def test_terminal_identity_checks_original_schema_without_admitting_inference(self):
        adapter, view = fixture(control=1, tick=6)
        view["suspended"] = True
        view["parts"][1]["detached"] = True
        self.assertEqual(adapter.terminal_identity(view).control_generation, 1)
        with self.assertRaises(ContractError):
            adapter.identity(view)
        for mutation in ("float_tick", "float_version", "parts", "geometry", "endpoint", "fault", "available"):
            changed = deepcopy(view)
            if mutation == "float_tick":
                changed["tick"] = 6.0
                changed["observation_id"] = "morphology-fixture:1:0:1:6.0"
            elif mutation == "float_version":
                changed["version"] = 2.0
            elif mutation == "parts":
                changed["parts"] = []
            elif mutation == "geometry":
                changed["parts"][0]["mass"] += .1
            elif mutation == "endpoint":
                changed["joints"][0]["b"] = 0
            elif mutation == "fault":
                changed["fault"] = "native actor failed"
            else:
                changed["suspended"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                adapter.terminal_identity(changed)

    def test_privileged_fields_and_identity_labels_do_not_enter_features(self):
        adapter, view = fixture()
        context = goal(view)
        original = adapter.frame(view, context)
        source = adapter.source_sha256(view)
        view.update(reward=10**9, success=True, evaluator={"target": [-900, -900]}, global_actors=[{"x": 0}],
                    cell_patch={"materials": [999]}, hidden_solver_state=123)
        self.assertEqual(adapter.frame(view, context).padded(), original.padded())
        self.assertEqual(adapter.source_sha256(view), source)
        view["run_id"] = "another-recorded-run"
        view["actor_id"] = 5
        view["observation_id"] = "another-recorded-run:1:5:0:0"
        changed = adapter.frame(view, context)
        self.assertEqual(changed.padded(), original.padded())
        self.assertEqual(changed.policy_sha256, original.policy_sha256)
        self.assertNotEqual(changed.identity, original.identity)

    def test_missing_is_distinct_from_present_stationary_zero(self):
        adapter, view = fixture("ripper")
        context = goal(view)
        view["grounded"] = False
        present = adapter.frame(view, context)
        view["grounded"] = None
        view["joints"][0]["travel"] = None
        missing = adapter.frame(view, context)
        self.assertEqual(present.root_values[0], missing.root_values[0])
        self.assertTrue(present.root_valid[0])
        self.assertFalse(missing.root_valid[0])
        i = EDGE_NAMES.index("travel/12")
        self.assertEqual(present.edges[0].values[i], 0)
        self.assertTrue(present.edges[0].valid[i])
        self.assertFalse(missing.edges[0].valid[i])
        self.assertNotEqual(present.padded(), missing.padded())

    def test_detached_part_stays_present_but_has_no_current_features_or_edges(self):
        adapter, view = fixture("ripper")
        before = adapter.frame(view, goal(view))
        view["parts"][2]["detached"] = True
        after = adapter.frame(view, goal(view))
        self.assertTrue(after.present[2])
        self.assertFalse(after.active[2])
        self.assertFalse(any(after.nodes[2]))
        self.assertFalse(after.edges[1].active)
        self.assertNotIn(2, information_reach(after, 8))
        self.assertNotEqual(before.identity.topology_sha256, after.identity.topology_sha256)
        with self.assertRaises(ContractError):
            replace(after.edges[1], values=before.edges[1].values, valid=before.edges[1].valid)

    def test_capability_mapping_preserves_distinct_button_meanings(self):
        request = SemanticAction(.8, .8, -.5, .7, True, True, True)
        expectations = {"jev": (True, True, {"drive_y", "brake"}),
                        "astra": (True, True, {"rebound"}),
                        "monowheel": (False, True, {"drive_y", "brake", "rebound"}),
                        "ripper": (False, True, {"drive_y", "impulse", "rebound"})}
        for name, (primary, secondary, masked) in expectations.items():
            adapter, view = fixture(name)
            action, labels = map_action(adapter, request, view, goal(view))
            with self.subTest(name=name):
                self.assertEqual((action.primary, action.secondary), (primary, secondary))
                self.assertEqual(set(labels), masked)
                self.assertLessEqual(sum(value * value for value in action.move), 1 + 1e-12)
                self.assertFalse(action.tool or action.auxiliary or action.release_usable or action.attach_usable)
                self.assertEqual(action.reel, 0)

    def test_full_profile_hash_and_actuator_semantics_are_fenced(self):
        profiles = strict_json_loads((ROOT / "configs/morphology/profiles-v1.json").read_bytes())
        row = profiles["profiles"]["jev"]
        row["definition"]["profile"]["primary"] = "spin"
        with self.assertRaisesRegex(ContractError, "morphology_profile_hash"):
            MorphologyAdapter(row["definition"], expected_sha256=row["profile_sha256"])
        with self.assertRaisesRegex(ContractError, "morphology_capability"):
            MorphologyAdapter(row["definition"], expected_sha256=sha256_json(row["definition"]))

    def test_geometry_bool_indices_unknown_parts_and_nonfinite_motion_refuse(self):
        mutations = [lambda v: v["parts"][0].update(index=True),
                     lambda v: v["parts"][0].update(mass=4.),
                     lambda v: v["parts"][0].update(id="foreign-body"),
                     lambda v: v["parts"][0].update(velocity=[float("nan"), 0]),
                     lambda v: v["joints"][0].update(a=True),
                     lambda v: v.update(tick=True),
                     lambda v: v.update(observation_scope="global"),
                     lambda v: v.update(observation_id="renewed"),
                     lambda v: v["parts"][0].update(detached=True)]
        for mutate in mutations:
            adapter, view = fixture()
            context = goal(view)
            mutate(view)
            with self.subTest(mutation=mutate), self.assertRaises(ContractError):
                adapter.frame(view, context)

    def test_original_age_and_observable_deadline_are_exclusive(self):
        adapter, view = fixture()
        context = goal(view)
        self.assertGreater(adapter.frame(view, context, current_tick=11).command[-1], 0)
        with self.assertRaisesRegex(ContractError, "morphology_expired"):
            adapter.frame(view, context, current_tick=12)
        adapter, view = fixture(tick=240)
        with self.assertRaisesRegex(ContractError, "morphology_deadline"):
            adapter.frame(view, context, started_tick=0)
        adapter, view = fixture(tick=234)
        frame = adapter.frame(view, context, started_tick=0, current_tick=239)
        self.assertAlmostEqual(frame.command[2], 1 / 240)
        with self.assertRaisesRegex(ContractError, "morphology_deadline"):
            adapter.frame(view, context, started_tick=0, current_tick=240)
        with self.assertRaises(ContractError):
            adapter.frame(view, context, source_generation=True)

    def test_state_is_independent_and_old_lifetime_cannot_resume(self):
        adapter, view = fixture()
        a, b = StateFence(), StateFence()
        context = goal(view)
        identity = adapter.identity(view)
        a.accept(identity, context)
        self.assertIsNone(b.identity)
        _, replacement = fixture(epoch=2)
        self.assertIn("actor_lifetime", a.accept(adapter.identity(replacement), context))
        with self.assertRaisesRegex(ContractError, "retired_morphology"):
            a.accept(identity, MorphologyGoal("different", context.target))

    def test_owner_mount_and_topology_change_need_a_fresh_invocation(self):
        for change in ("source", "control", "mount", "part"):
            adapter, view = fixture()
            fence, context = StateFence(), goal(view)
            fence.accept(adapter.identity(view), context)
            _, following = fixture(tick=6, control=1 if change == "control" else 0)
            if change == "part":
                following["parts"][1]["detached"] = True
            identity = adapter.identity(following, source_generation=2 if change == "source" else 1,
                                        mount_generation=1 if change == "mount" else None)
            with self.subTest(change=change), self.assertRaisesRegex(ContractError, "retired_invocation"):
                fence.accept(identity, context)
            self.assertIn("goal_invocation", fence.accept(identity, replace(context, id="fresh")))

    def test_old_authority_and_reordered_ticks_do_not_reenter_after_cancel(self):
        adapter, view = fixture(tick=18, control=2)
        fence, context = StateFence(), goal(view)
        fence.accept(adapter.identity(view, source_generation=2, mount_generation=2), context)
        fence.cancel()
        for tick, control, source, mount, code in ((12, 2, 2, 2, "morphology_reordered"),
                                                   (24, 1, 2, 2, "retired_owner"),
                                                   (24, 2, 1, 2, "retired_owner"),
                                                   (24, 2, 2, 1, "retired_mount")):
            _, old = fixture(tick=tick, control=control)
            identity = adapter.identity(old, source_generation=source, mount_generation=mount)
            with self.subTest(code=code), self.assertRaisesRegex(ContractError, code):
                fence.accept(identity, replace(context, id="fresh"))

    def test_goal_collision_and_cadence_skip_retire_the_invocation(self):
        adapter, view = fixture()
        fence, context = StateFence(), goal(view)
        fence.accept(adapter.identity(view), context)
        _, following = fixture(tick=6)
        with self.assertRaisesRegex(ContractError, "morphology_goal_collision"):
            fence.accept(adapter.identity(following), replace(context, target=(0., 0.)))
        with self.assertRaisesRegex(ContractError, "retired_invocation"):
            fence.accept(adapter.identity(following), context)
        fence.accept(adapter.identity(following), replace(context, id="second"))
        _, gap = fixture(tick=18)
        with self.assertRaisesRegex(ContractError, "morphology_cadence"):
            fence.accept(adapter.identity(gap), replace(context, id="second"))

    def test_unavailable_mount_sample_cannot_erase_the_known_generation(self):
        adapter, view = fixture()
        fence, context = StateFence(), goal(view)
        fence.accept(adapter.identity(view, mount_generation=2), context)
        _, following = fixture(tick=6)
        with self.assertRaisesRegex(ContractError, "morphology_mount_unavailable"):
            fence.accept(adapter.identity(following, mount_generation=None), replace(context, id="unknown"))
        with self.assertRaisesRegex(ContractError, "retired_mount"):
            fence.accept(adapter.identity(following, mount_generation=1), replace(context, id="older"))
        self.assertEqual(fence.identity.mount_generation, 2)
        _, reset = fixture(epoch=2)
        self.assertIn("actor_lifetime", fence.accept(adapter.identity(reset), context))
        self.assertIsNone(fence.identity.mount_generation)


class MorphologyAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.config = strict_json_loads((ROOT / "configs/morphology/comparison-v1.json").read_bytes())
        self.declaration = strict_json_loads((ROOT / "configs/morphology/entry-gate-v1.json").read_bytes())
        self.result = strict_json_loads((ROOT / "docs/evidence/t106-prerequisite-result.json").read_bytes())
        self.profiles = strict_json_loads((ROOT / "configs/morphology/profiles-v1.json").read_bytes())

    def test_actual_t105_result_keeps_both_families_closed(self):
        result = entry_decision(self.declaration, self.result)
        self.assertFalse(result["admitted"])
        self.assertIsNone(result["selected_family"])
        for family in ("bc", "residual"):
            self.assertEqual(set(result["candidates"][family]["failed"]),
                             {"minimum_mean_success", "minimum_every_seed_success"})

    def test_relaxed_gate_or_changed_evidence_is_refused(self):
        for mutation in ("threshold", "result", "missing_seed_family"):
            declaration, result = deepcopy(self.declaration), deepcopy(self.result)
            if mutation == "threshold":
                declaration["minimum_mean_success"] = .4
                declaration["minimum_every_seed_success"] = .2
            elif mutation == "result":
                result["mechanical_utility_prerequisite_passed"] = True
            else:
                del result["candidates"]["bc"]
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                entry_decision(declaration, result)

    def test_six_fixed_jobs_whole_arrangements_geometry_and_budget_are_explicit(self):
        self.assertEqual(sha256_json(self.config), COMPARISON_SHA256)
        plan = comparison_plan(self.config, self.profiles)
        self.assertEqual(len(plan["jobs"]), 6)
        self.assertEqual({row["seed"] for row in plan["jobs"]}, {6113, 6229, 6337})
        self.assertEqual(plan["allocated_phase_ceiling_sum"], 12704)
        self.assertLessEqual(plan["allocated_phase_ceiling_sum"], 13000)
        self.assertEqual(len(set(plan["arrangement_fingerprints"].values())), 4)
        self.assertEqual(len(set(plan["geometry_fingerprints"].values())), 4)
        self.assertFalse(plan["current_training_authorized"])
        self.assertFalse(plan["future_trainer_backend_implemented_or_admitted"])

    def test_blocked_run_is_immutable_and_counts_no_work(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "blocked"
            kwargs = dict(root=ROOT, config_path=ROOT / "configs/morphology/comparison-v1.json",
                          declaration_path=ROOT / "configs/morphology/entry-gate-v1.json",
                          result_path=ROOT / "docs/evidence/t106-prerequisite-result.json",
                          profiles_path=ROOT / "configs/morphology/profiles-v1.json", output=output)
            result = prepare_comparison(**kwargs)
            self.assertEqual(result["status"], "training_blocked", result["errors"])
            self.assertFalse(result["admitted"])
            self.assertTrue(all(count == 0 for count in result["work"].values()))
            data = (output / "admission.json").read_bytes()
            with self.assertRaises(FileExistsError):
                prepare_comparison(**kwargs)
            self.assertEqual((output / "admission.json").read_bytes(), data)

    def test_missing_evidence_is_inconclusive_with_zero_work(self):
        with tempfile.TemporaryDirectory() as directory:
            result = prepare_comparison(root=ROOT, config_path=ROOT / "configs/morphology/comparison-v1.json",
                                        declaration_path=ROOT / "configs/morphology/entry-gate-v1.json",
                                        result_path=Path(directory) / "absent-result.json",
                                        profiles_path=ROOT / "configs/morphology/profiles-v1.json",
                                        output=Path(directory) / "invalid")
            self.assertEqual(result["status"], "evidence_invalid")
            self.assertFalse(result["admitted"])
            self.assertTrue(result["errors"])
            self.assertTrue(all(count == 0 for count in result["work"].values()))

    def test_cli_refusal_precedes_torch_and_host_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            source = """import runpy,sys
sys.argv=['run.py','--output',sys.argv[1],'--expect-blocked']
try:
    runpy.run_path('evals/morphology/run.py',run_name='__main__')
except SystemExit as exc:
    assert exc.code==0
assert 'torch' not in sys.modules
assert 'nca.envs' not in sys.modules
assert 'nca.controllers.morphology.models' not in sys.modules
"""
            environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
            run = subprocess.run([sys.executable, "-c", source, str(Path(directory) / "closed")], cwd=ROOT,
                                 capture_output=True, text=True, env=environment)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout)["status"], "training_blocked")


class MorphologyConformanceTests(unittest.TestCase):
    def test_startup_recovery_preserves_failed_attempt_and_original_total_budget(self):
        import hashlib
        from nca.controllers.morphology.conformance import recovery_plan, _error
        declaration = strict_json_loads((ROOT / "configs/morphology/conformance-recovery-v1.json").read_bytes())
        data = (ROOT / "docs/evidence/t106-conformance-startup-failed.json").read_bytes()
        previous = strict_json_loads(data)
        digest = hashlib.sha256(data).hexdigest()
        plan = recovery_plan(declaration, previous, previous_file_sha256=digest)
        self.assertLessEqual(plan["previous_wall_seconds"] + plan["remaining_wall_seconds"], 60)
        self.assertEqual(plan["maximum_aggregate_game_decisions"], 32)
        self.assertFalse(plan["automatic_retry"])
        changed = deepcopy(previous)
        changed["budget"]["attempted_decisions"] = 1
        for candidate, record, file_hash in ((dict(declaration, maximum_remaining_wall_seconds=60), previous, digest),
                                              (declaration, changed, digest), (declaration, previous, "0" * 64)):
            with self.assertRaises(ContractError):
                recovery_plan(candidate, record, previous_file_sha256=file_hash)
        from nca.adapters.protocol import HostError
        try:
            try:
                raise TimeoutError("private path or transport detail")
            except TimeoutError as cause:
                raise HostError("persistent host did not start: private path") from cause
        except HostError as exc:
            error = _error(exc)
        self.assertEqual(error["cause_type"], "TimeoutError")
        self.assertNotIn("private path", str(error))

    def test_uncertain_rpc_keeps_its_reserved_decision_and_wall_limit_stops_more_work(self):
        from nca.controllers.morphology.conformance import ConformanceBudget
        with patch("nca.controllers.morphology.conformance.time.monotonic", return_value=100.):
            budget = ConformanceBudget(maximum_decisions=1)
            budget.reserve()
            self.assertEqual(budget.receipt()["unconfirmed_decisions"], 1)
            with self.assertRaisesRegex(ContractError, "morphology_conformance_decisions"):
                budget.reserve()
        with patch("nca.controllers.morphology.conformance.time.monotonic", return_value=160.):
            with self.assertRaisesRegex(ContractError, "morphology_conformance_wall"):
                budget.check()
        self.assertEqual(budget.attempted_decisions, 1)

    def test_consumed_replacement_and_source_restamping_do_not_pass_conformance(self):
        from nca.controllers.morphology.conformance import validate_interval
        from nca.contracts import WireAction
        packet = {"run_id": "fixture", "epoch": 1, "actor_id": 0, "sequence": 0,
                  "observation_id": "fixture:1:0:0:0", "observed_tick": 0, "lease_ticks": 12,
                  "action": WireAction(move=(.35, 0), effort=.6).to_dict()}
        info = {"actual_ticks": 6, "elapsed_ticks": 6, "delay_ticks_before_repeat": 0,
                "consumed_trace_complete": True, "trace": []}
        for i in range(6):
            info["trace"].append({"tick_before": i, "tick_after": i + 1, "failed": False,
                                   "admission": [{"accepted": True}] if i == 0 else [],
                                   "requested": [deepcopy(packet)] if i == 0 else [],
                                   "applied": {"action_available": True, "terminal_revocation": False,
                                               "action": deepcopy(packet["action"])}})
        validate_interval(packet, {"tick": 0}, {"tick": 6}, False, False, info)
        for mutation in ("source", "request", "replacement", "unavailable", "float_boolean"):
            changed = deepcopy(info)
            if mutation == "source":
                changed["trace"][0]["requested"][0]["observation_id"] = "new-view"
            elif mutation == "request":
                changed["trace"][0]["requested"][0]["action"]["move"][0] = -.35
            elif mutation == "replacement":
                changed["trace"][3]["applied"]["action"]["move"][0] = -.35
            elif mutation == "unavailable":
                changed["trace"][4]["applied"]["action_available"] = False
            else:
                changed["trace"][2]["applied"]["action"]["primary"] = 0.
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                validate_interval(packet, {"tick": 0}, {"tick": 6}, False, False, changed)

    def test_cancellation_preserves_physics_and_revokes_targets_and_authority(self):
        from nca.controllers.morphology.conformance import cancellation_receipt
        _, view = fixture()
        view["joints"][0]["target"] = .5
        after = deepcopy(view)
        after["joints"][0]["target"] = 0.
        after["control_generation"] += 1
        after["observation_id"] = ":".join(str(after[key]) for key in
                                             ("run_id", "epoch", "actor_id", "control_generation", "tick"))
        receipt = cancellation_receipt(view, after)
        self.assertTrue(receipt["same_tick_pose_velocity_parts_and_joint_physics"])
        self.assertEqual(receipt["joint_targets_before"][0], .5)
        self.assertTrue(receipt["joint_targets_revoked"])
        for mutation in ("tick", "part", "authority", "stamp", "joint", "target", "nonfinite"):
            changed = deepcopy(after)
            if mutation == "tick":
                changed["tick"] += 1
            elif mutation == "part":
                changed["parts"][1]["position"][0] += .01
            elif mutation == "stamp":
                changed["observation_id"] = view["observation_id"]
            elif mutation == "joint":
                changed["joints"][0]["angle"] += .01
            elif mutation == "target":
                changed["joints"][0]["target"] = .5
            elif mutation == "nonfinite":
                changed["joints"][0]["target"] = float("nan")
            else:
                changed["control_generation"] -= 1
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                cancellation_receipt(view, changed)


if __name__ == "__main__":
    unittest.main()
