"""Synthetic receipt/accounting tests, never physical reference qualification."""

from copy import deepcopy
from dataclasses import replace
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

from nca.contracts import ContractError, WireAction, sha256_json, strict_json_loads
from nca.controllers.morphology import MorphologyAdapter, MorphologyGoal, SemanticAction, map_action
from nca.train.morphology_task import (
    REFERENCE_SPEC, TASK_SPEC, WaypointReference, WaypointTask,
    evaluate_qualification, qualification_cases,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILES = strict_json_loads((ROOT / "configs/morphology/profiles-v1.json").read_bytes())


def fixture(model="jev"):
    row = PROFILES["profiles"][model]
    adapter = MorphologyAdapter(row["definition"], expected_sha256=row["profile_sha256"])
    view = deepcopy(row["observation"])
    view.update(run_id="synthetic-waypoint-receipts", tick=0, epoch=1, control_generation=0, grounded=True)
    stamp(view)
    return adapter, view


def stamp(view):
    view["observation_id"] = ":".join(str(view[key]) for key in
                                     ("run_id", "epoch", "actor_id", "control_generation", "tick"))
    return view


def goal(view, dx=48., name="fixture-goal"):
    x, y = view["parts"][0]["position"]
    return MorphologyGoal(name, (x + dx, y))


def interval(before, action=None, *, sequence=None, sample=None, terminal=False):
    """Construct labelled test receipts; this is not a physics model or replay."""
    action = WireAction(effort=0.) if action is None else action
    sequence = before["tick"] // 6 + 1 if sequence is None else sequence
    packet = {"version": 2, "kind": "physical_wire_action", "run_id": before["run_id"],
              "actor_id": before["actor_id"], "epoch": before["epoch"], "sequence": sequence,
              "observation_id": before["observation_id"], "observed_tick": before["tick"],
              "lease_ticks": min(12, 240 - before["tick"]), "action": action.to_dict()}
    trace = []
    for offset in range(6):
        tick = before["tick"] + offset + 1
        view = deepcopy(before)
        view["tick"] = tick
        if sample is not None:
            sample(tick, view)
        last = terminal and offset == 5
        if last:
            view["suspended"] = True
            view["control_generation"] += 1
        stamp(view)
        trace.append({"tick_before": tick - 1, "tick_after": tick, "world_tick": tick,
                      "cellular_sample_tick": tick - 1, "failed": False,
                      "received_monotonic_usec": tick * 1000, "completed_monotonic_usec": tick * 1000 + 900,
                      "admission": [{"accepted": True, "reason": ""}] if offset == 0 else [],
                      "requested": [deepcopy(packet)] if offset == 0 else [], "observation": view,
                      "applied": {"action_available": not last, "terminal_revocation": last,
                                  "action": None if last else action.to_dict()}})
    return action, trace, deepcopy(trace[-1]["observation"])


def qualification_rows():
    """Complete fabricated rows solely for the conservative gate's unit tests."""
    result = []
    for case in qualification_cases(PROFILES):
        result.append({"scenario_family": case["id"], "profile": case["profile"], "host_seed": case["host_seed"],
                       "phase": "qualification", "kind": "reference", "decisions": 2, "completed_ticks": 12,
                       "policy_sha256": sha256_json(REFERENCE_SPEC), "task_spec_sha256": sha256_json(TASK_SPEC),
                       "artifact_sha256": "a" * 64, "record_sha256": "b" * 64, "boundary_audit_sha256": "c" * 64,
                       "profile_sha256": PROFILES["profiles"][case["profile"]]["profile_sha256"],
                       "source_boundary_regressions": 0, "actual_owner_closed": True, "error": None,
                       "metrics": {"success": True, "reason": "arrival", "administratively_capped": False,
                                   "integrity_failure": False, "owner_feel_accepted": False, "failed_admissions": 0,
                                   "elapsed_ticks": 12, "success_seconds": .2, "capped_completion_seconds": .2,
                                   "inference_p50_ms": 1., "inference_p95_ms": 1.1,
                                   "ordinary_trim_reversals_per_second": 0., "mean_absolute_angle_radians": .1,
                                   "peak_useful_speed_cells_per_second": 15.}})
    return result


class WaypointReferenceTests(unittest.TestCase):
    def test_module_import_needs_neither_torch_nor_physical_host(self):
        code = ("import sys; import nca.train.morphology_task; "
                "assert 'torch' not in sys.modules; assert 'nca.adapters.godot' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_frozen_task_has_distinct_reward_and_inclusive_comparison_arrival(self):
        self.assertNotEqual(TASK_SPEC["reward"]["schema"], "jolt-arrival-best-progress/v1")
        self.assertEqual(TASK_SPEC["arrival"]["absolute_angle_maximum_radians"], .35)
        self.assertTrue(TASK_SPEC["arrival"]["deadline_inclusive"])
        self.assertEqual(TASK_SPEC["qualification"]["maximum_total_requests"], 160)
        adapter, view = fixture()
        for changed in (replace(goal(view), tolerance=4.01), replace(goal(view), deadline_ticks=246)):
            with self.subTest(changed=changed), self.assertRaises(ContractError):
                WaypointTask(adapter, changed, view)

    def test_bounded_servo_is_symmetric_and_uses_permitted_numeric_speed(self):
        adapter, view = fixture()
        scale = min(240., max(5., adapter.profile["profile"]["speed"] * .8))
        east = WaypointReference(adapter).step(view, goal(view, 26))
        west = WaypointReference(adapter).step(view, goal(view, -26))
        self.assertAlmostEqual(east.drive_x, min(30., scale) / scale)
        self.assertEqual(east.drive_x, -west.drive_x)
        self.assertEqual((east.drive_y, east.effort, east.impulse, east.rebound), (0., .8, False, False))
        for speed in (1., 1000.):
            definition = deepcopy(PROFILES["profiles"]["jev"]["definition"])
            definition["profile"]["speed"] = speed
            changed = MorphologyAdapter(definition, expected_sha256=sha256_json(definition))
            request = WaypointReference(changed).step(view, goal(view, 26))
            denominator = max(5., min(240., speed * .8))
            self.assertAlmostEqual(request.drive_x, min(30., denominator) / denominator)

    def test_root_and_goal_determine_steering_while_rig_and_evaluator_values_do_not(self):
        adapter, view = fixture()
        context = goal(view)
        expected = WaypointReference(adapter).step(view, context)
        view.update(position=[999, -999], velocity=[-900, 500], rotation=2., angular_velocity=9.,
                    evaluator={"target": [-1000, -1000]}, reward=10**8, success=True, global_actors=[1])
        self.assertEqual(WaypointReference(adapter).step(view, context), expected)
        view["parts"][0]["velocity"] = [30., 0.]
        self.assertLess(WaypointReference(adapter).step(view, context).drive_x, expected.drive_x)

    def test_flight_vertical_drive_brake_and_trim_obey_capability_and_strict_edges(self):
        adapter, view = fixture("astra")
        context = goal(view, 0)
        view["parts"][0].update(velocity=[3., 0.], rotation=1., angular_velocity=8.)
        action = WaypointReference(adapter).step(view, context)
        self.assertLess(action.drive_x, 0.)
        self.assertTrue(action.brake)
        self.assertEqual(action.trim, -1.)
        self.assertFalse(action.impulse or action.rebound)
        view["parts"][0]["velocity"] = [2., 0.]
        self.assertFalse(WaypointReference(adapter).step(view, context).brake)
        view["parts"][0]["velocity"] = [-3., 0.]
        self.assertTrue(WaypointReference(adapter).step(view, goal(view, 26)).brake)
        target = replace(goal(view), target=(view["parts"][0]["position"][0], view["parts"][0]["position"][1] - 20.))
        vertical = WaypointReference(adapter).step(view, target)
        self.assertLess(vertical.drive_y, 0.)
        wire, masked = map_action(adapter, vertical, view, target)
        self.assertEqual(masked, ())
        self.assertLessEqual(math.hypot(*wire.move), 1.)
        self.assertFalse(wire.tool or wire.auxiliary or wire.reel or wire.release_usable or wire.attach_usable)
        ground_adapter, ground = fixture()
        ground["parts"][0]["velocity"] = [3., 100.]
        ground_action = WaypointReference(ground_adapter).step(ground, goal(ground, 0))
        self.assertEqual(ground_action.drive_y, 0.)
        self.assertFalse(ground_action.brake)

    def test_duplicate_source_is_cached_but_changed_bytes_retire_the_invocation(self):
        adapter, view = fixture()
        reference, context = WaypointReference(adapter), goal(view)
        first = reference.step(view, context)
        self.assertIs(reference.step(deepcopy(view), context), first)
        self.assertEqual(reference.previous, first)
        view["parts"][1]["velocity"][0] += .1
        with self.assertRaisesRegex(ContractError, "waypoint_source_changed"):
            reference.step(view, context)
        self.assertEqual(reference.previous, SemanticAction())
        self.assertIsNone(reference._cache)
        with self.assertRaisesRegex(ContractError, "retired_invocation"):
            reference.step(view, context)

    def test_trim_retains_the_declared_half_open_wrapped_angle_at_pi(self):
        adapter, view = fixture()
        for angle, expected in ((math.pi, 1.), (-math.pi, 1.), (.35, -.56)):
            view["parts"][0].update(rotation=angle, angular_velocity=0.)
            request = WaypointReference(adapter).step(view, goal(view))
            self.assertAlmostEqual(request.trim, expected)

    def test_cancellation_does_not_erase_source_or_lifetime_history(self):
        adapter, initial = fixture()
        for method in ("cancel", "reset"):
            reference, context = WaypointReference(adapter), goal(initial)
            reference.step(initial, context)
            getattr(reference, method)()
            with self.assertRaisesRegex(ContractError, "retired_invocation"):
                reference.step(initial, context)
            fresh = replace(context, id="fresh-invocation")
            reference.step(initial, fresh)
            next_life = deepcopy(initial)
            next_life["run_id"] = "another-synthetic-lifetime"
            stamp(next_life)
            reference.step(next_life, context)
            with self.assertRaisesRegex(ContractError, "retired_morphology"):
                reference.step(initial, context)

    def test_authority_goal_cadence_expiry_and_nonfinite_sources_fail_closed(self):
        for mutation in ("control", "source", "topology", "goal", "cadence", "expired", "nan", "boolean"):
            adapter, view = fixture()
            reference, context = WaypointReference(adapter), goal(view)
            reference.step(view, context)
            kwargs = {}
            if mutation == "control":
                view.update(tick=6, control_generation=1)
            elif mutation == "source":
                view["tick"] = 6
                kwargs["source_generation"] = 2
            elif mutation == "topology":
                view["tick"] = 6
                view["joints"][0]["enabled"] = False
            elif mutation == "goal":
                view["tick"] = 6
                context = replace(context, target=(context.target[0] + 1, context.target[1]))
            elif mutation == "cadence":
                view["tick"] = 12
            elif mutation == "expired":
                kwargs["current_tick"] = 12
            elif mutation == "nan":
                view["parts"][0]["velocity"][0] = float("nan")
            else:
                view["parts"][0]["velocity"][0] = True
            stamp(view)
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                reference.step(view, context, **kwargs)
            self.assertEqual(reference.previous, SemanticAction())
        adapter, view = fixture()
        reference, context = WaypointReference(adapter), goal(view)
        for tick in range(0, 240, 6):
            view["tick"] = tick
            reference.step(stamp(view), context)
        view["tick"] = 240
        with self.assertRaisesRegex(ContractError, "morphology_deadline"):
            reference.step(stamp(view), context)


class WaypointTaskTests(unittest.TestCase):
    def test_exact_twelve_tick_quiet_arrival_uses_root_and_does_not_require_support(self):
        adapter, view = fixture()
        context = goal(view, 0.)
        task = WaypointTask(adapter, context, view)
        def sample(tick, observation):
            observation["parts"][0].update(position=[context.target[0] - 4., context.target[1]],
                                             velocity=[6., 0.], rotation=.35)
            observation.update(grounded=None, position=[999, 999], velocity=[999, 999], rotation=2.)
        total = 0.
        for i in range(2):
            action, trace, view = interval(view, sample=sample)
            terms = task.consume(action, trace, inference_ns=1_000_000)
            total += terms["arrival"]
            self.assertEqual(task.succeeded, i == 1)
        self.assertEqual(total, 1.)
        self.assertEqual((task.elapsed_ticks, task.success_tick), (12, 12))
        self.assertEqual(task.metrics()["capped_completion_seconds"], .2)
        with self.assertRaisesRegex(ContractError, "waypoint_finished"):
            task.consume(action, trace)

    def test_interrupted_quiet_count_resets_and_remainder_of_actual_repeat_is_counted(self):
        adapter, view = fixture()
        context = goal(view, 0.)
        task = WaypointTask(adapter, context, view)
        def sample(tick, observation):
            observation["parts"][0].update(velocity=[6.001 if tick == 11 else 0., 0.])
        for _ in range(4):
            action, trace, view = interval(view, sample=sample)
            task.consume(action, trace)
        self.assertEqual(task.success_tick, 23)
        self.assertEqual(task.elapsed_ticks, 24)
        self.assertAlmostEqual(task.reward_totals["time"], -.01 * 24 / 60)
        self.assertEqual(task.reward_totals["arrival"], 1.)

    def test_final_deadline_tick_arrival_precedes_penalty_and_preserves_revocation(self):
        for arrives in (True, False):
            adapter, view = fixture()
            context = goal(view, 0.)
            task = WaypointTask(adapter, context, view)
            def sample(tick, observation):
                observation["parts"][0].update(position=[context.target[0] + (0. if arrives and tick >= 229 else 10.),
                                                         context.target[1]], velocity=[0., 0.], rotation=0.)
            for i in range(40):
                action, trace, view = interval(view, sample=sample, terminal=i == 39)
                task.consume(action, trace)
            with self.subTest(arrives=arrives):
                self.assertEqual(task.succeeded, arrives)
                self.assertEqual(task.deadline_missed, not arrives)
                self.assertEqual(task.reward_totals["deadline"], 0. if arrives else -.5)
                self.assertEqual(task.reward_totals["arrival"], float(arrives))
                self.assertEqual(task.elapsed_ticks, 240)
                self.assertEqual(task.metrics()["capped_completion_seconds"], 4.)
                self.assertFalse(trace[-1]["applied"]["action_available"])
                self.assertIsNone(trace[-1]["applied"]["action"])

    def test_closed_path_has_zero_potential_and_idle_has_only_declared_costs(self):
        adapter, view = fixture("astra")
        context = goal(view)
        task = WaypointTask(adapter, context, view)
        distances = (45., 42., 40., 42., 45., 48.)
        def sample(tick, observation):
            observation["parts"][0]["position"] = [context.target[0] - distances[tick - 1], context.target[1]]
        action = WireAction(move=(.2, .3), trim=.4, effort=.6)
        _, trace, _ = interval(view, action, sample=sample)
        terms = task.consume(action, trace)
        self.assertAlmostEqual(terms["progress"], 0.)
        self.assertAlmostEqual(terms["time"], -.001)
        self.assertAlmostEqual(terms["request_cost"], -.002 * (.04 + .09 + .16 + .36) * .1)
        self.assertLess(terms["total"], 0.)
        idle = WaypointTask(adapter, context, view)
        action, trace, _ = interval(view)
        self.assertAlmostEqual(idle.consume(action, trace)["total"], -.001)

    def test_receipt_reordering_mixed_sources_refusals_and_missing_reads_never_reward(self):
        for mutation in ("gap", "world", "cellular", "failed", "request_source", "request_action", "missing_admission",
                         "refused", "replacement", "owner", "topology", "action", "unavailable", "clock", "nan", "bool_tick"):
            adapter, view = fixture()
            task = WaypointTask(adapter, goal(view), view)
            action, trace, _ = interval(view)
            row = trace[2]
            if mutation == "gap": row["tick_before"] += 1
            elif mutation == "world": row["world_tick"] -= 1
            elif mutation == "cellular": row["cellular_sample_tick"] += 1
            elif mutation == "failed": row["failed"] = True
            elif mutation == "request_source": trace[0]["requested"][0]["observation_id"] = "refreshed-original-source"
            elif mutation == "request_action": trace[0]["requested"][0]["action"]["trim"] = .1
            elif mutation == "missing_admission": trace[0]["admission"] = []
            elif mutation == "refused": trace[0]["admission"][0]["accepted"] = False
            elif mutation == "replacement": row["requested"] = [deepcopy(trace[0]["requested"][0])]
            elif mutation == "owner":
                row["observation"]["control_generation"] += 1
                stamp(row["observation"])
            elif mutation == "topology": row["observation"]["parts"][1]["detached"] = True
            elif mutation == "action": row["applied"]["action"]["trim"] = .1
            elif mutation == "unavailable": row["applied"].update(action_available=False, action=None)
            elif mutation == "clock": row["completed_monotonic_usec"] = 0
            elif mutation == "nan": row["observation"]["parts"][0]["position"][0] = float("nan")
            else: trace[0]["tick_before"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                task.consume(action, trace)
            self.assertTrue(task.integrity_failure)
            self.assertEqual(task.elapsed_ticks, 0)
            self.assertEqual(sum(task.reward_totals.values()), 0.)
            self.assertEqual(task.inference_ns, [])
            self.assertEqual(task.failed_admissions, int(mutation == "refused"))

    def test_partial_or_early_terminal_and_extra_capability_are_invalid(self):
        for mutation in ("empty", "partial", "early_terminal", "tool", "vertical", "inference_bool"):
            adapter, view = fixture()
            task = WaypointTask(adapter, goal(view), view)
            action, trace, _ = interval(view, terminal=mutation == "early_terminal")
            if mutation == "empty": trace = []
            elif mutation == "partial": trace = trace[:3]
            elif mutation == "tool": action = replace(action, tool=True)
            elif mutation == "vertical": action = replace(action, move=(0., .1))
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                task.consume(action, trace, inference_ns=True if mutation == "inference_bool" else 0)
            self.assertTrue(task.integrity_failure)

    def test_float_readback_tolerance_is_finite_and_sequence_is_consecutive(self):
        for change, accepted in ((5e-7, True), (2e-6, False), (float("nan"), False), (True, False)):
            adapter, view = fixture()
            task = WaypointTask(adapter, goal(view), view)
            action, trace, _ = interval(view)
            trace[2]["applied"]["action"]["trim"] = change
            if accepted:
                task.consume(action, trace)
            else:
                with self.assertRaises(ContractError): task.consume(action, trace)
        adapter, view = fixture()
        task = WaypointTask(adapter, goal(view), view)
        action, trace, view = interval(view, sequence=3)
        task.consume(action, trace)
        action, trace, _ = interval(view, sequence=3)
        with self.assertRaisesRegex(ContractError, "waypoint_request"):
            task.consume(action, trace)
        self.assertEqual(task.elapsed_ticks, 6)

    def test_native_request_echo_integral_floats_preserve_original_integer_source(self):
        adapter, view = fixture()
        task = WaypointTask(adapter, goal(view), view)
        for _ in range(2):
            action, trace, view = interval(view)
            packet = trace[0]["requested"][0]
            for field in ("version", "epoch", "actor_id", "observed_tick", "lease_ticks", "sequence"):
                packet[field] = float(packet[field])
            task.consume(action, trace)
        self.assertEqual(task.elapsed_ticks, 12)
        for field, value in (("epoch", True), ("epoch", 1.1), ("sequence", 1.1),
                             ("sequence", True), ("sequence", float("nan")), ("observed_tick", float("inf"))):
            adapter, view = fixture()
            task = WaypointTask(adapter, goal(view), view)
            action, trace, _ = interval(view)
            trace[0]["requested"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ContractError):
                task.consume(action, trace)
            self.assertTrue(task.integrity_failure)

    def test_actual_metrics_use_all_ticks_deadband_and_linear_inference_quantiles(self):
        adapter, view = fixture()
        task = WaypointTask(adapter, goal(view), view)
        def sample(tick, observation):
            observation["parts"][0].update(velocity=[3. if tick == 1 else 10., 0.], rotation=.2)
        for trim, latency in zip((.1, 0., -.1, .049, .1), (1, 4, 2, 3, 5)):
            action, trace, view = interval(view, WireAction(trim=trim), sample=sample)
            task.consume(action, trace, inference_ns=latency * 1_000_000)
        metrics = task.metrics()
        self.assertEqual(metrics["elapsed_ticks"], 30)
        self.assertEqual(metrics["ordinary_trim_reversals"], 2)
        self.assertEqual(metrics["ordinary_trim_reversals_per_second"], 4.)
        self.assertAlmostEqual(metrics["mean_absolute_angle_radians"], .2)
        self.assertEqual(metrics["peak_useful_speed_cells_per_second"], 10.)
        self.assertEqual(metrics["response_seconds"], 1 / 60)
        self.assertEqual(metrics["inference_p50_ms"], 3.)
        self.assertAlmostEqual(metrics["inference_p95_ms"], 4.8)
        self.assertIsNone(metrics["settled_after_landing_seconds"])
        self.assertFalse(metrics["owner_feel_accepted"])

    def test_landing_needs_actual_loss_reacquisition_and_twelve_supported_settled_ticks(self):
        for initial_support, lands, expected in ((True, True, 11 / 60), (False, True, None), (True, False, None), (None, True, None)):
            adapter, view = fixture()
            view["grounded"] = initial_support
            task = WaypointTask(adapter, goal(view), view)
            def sample(tick, observation):
                observation["grounded"] = tick >= 3 if lands else False
                observation["parts"][0].update(velocity=[0., 0.], rotation=0.)
            for _ in range(3):
                action, trace, view = interval(view, sample=sample)
                task.consume(action, trace)
            self.assertEqual(task.metrics()["settled_after_landing_seconds"], expected)

    def test_administrative_cutoff_is_explicit_and_never_invents_deadline_cost(self):
        adapter, view = fixture()
        task = WaypointTask(adapter, goal(view), view)
        self.assertIsNone(task.metrics()["inference_p95_ms"])
        self.assertIsNone(task.metrics()["mean_absolute_angle_radians"])
        self.assertEqual(task.metrics()["reason"], "incomplete")
        action, trace, _ = interval(view)
        task.consume(action, trace)
        metrics = task.metrics(administratively_capped=True)
        self.assertEqual(metrics["reason"], "administrative_cutoff")
        self.assertFalse(metrics["success"])
        self.assertEqual(metrics["capped_completion_seconds"], 4.)
        self.assertEqual(metrics["reward_terms"]["deadline"], 0.)


class WaypointQualificationTests(unittest.TestCase):
    def test_cases_use_fresh_geometry_training_profiles_and_distinct_actor_origin(self):
        rows = qualification_cases(PROFILES)
        self.assertEqual([row["host_seed"] for row in rows], [610651, 610652, 610653, 610654])
        self.assertEqual([row["spawn"][0] for row in rows], [264., 176., 264., 176.])
        self.assertEqual([row["target"][0] - row["spawn"][0] for row in rows], [-26., 26., -26., 26.])
        self.assertEqual([row["target"][1] for row in rows], [346.5, 346.5, 346., 346.])
        self.assertTrue(all(set(row) == {"id", "profile", "spawn", "target", "host_seed"} for row in rows))
        only_training = {"profiles": {key: PROFILES["profiles"][key] for key in ("jev", "astra")}}
        self.assertEqual(qualification_cases(only_training), rows)
        changed = deepcopy(only_training)
        changed["profiles"]["jev"]["observation"]["parts"][0]["position"][1] += 1.5
        moved = qualification_cases(changed)
        self.assertEqual(moved[0]["spawn"], rows[0]["spawn"])
        self.assertEqual(moved[0]["target"][1], rows[0]["target"][1] + 1.5)

    def test_full_synthetic_rows_exercise_gate_without_claiming_artifact_validation(self):
        result = evaluate_qualification(qualification_rows())
        self.assertTrue(result["complete"])
        self.assertTrue(result["passed"])
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["maximum_game_requests"], 160)
        self.assertIn("caller must verify original artifact bytes", result["evidence_limit"])
        self.assertEqual(len(result["checks"]), 4)

    def test_missing_duplicate_foreign_or_malformed_rows_keep_gate_closed(self):
        valid = qualification_rows()
        for rows in ([], None, valid[:3], valid + [deepcopy(valid[0])], [*valid[:3], {}],
                     [*valid[:3], None], [*valid[:3], {"scenario_family": []}]):
            with self.subTest(rows=rows):
                result = evaluate_qualification(rows)
                self.assertFalse(result["complete"])
                self.assertFalse(result["passed"])
        duplicate = deepcopy(valid)
        duplicate[-1] = deepcopy(duplicate[0])
        self.assertFalse(evaluate_qualification(duplicate)["passed"])

    def test_one_invalid_row_cannot_be_averaged_away_or_counted_as_convenient_success(self):
        for mutation in ("seed", "seed_bool", "profile", "kind", "phase", "task_hash", "reference_hash", "source_hash",
                         "source", "error", "owner", "cap", "integrity", "admission", "success_bool", "owner_feel",
                         "latency", "missing_latency", "nan", "reversed_quantiles", "ticks", "decisions_bool",
                         "task_ticks", "success_time", "completion_time", "reason"):
            rows = qualification_rows()
            row, metrics = rows[0], rows[0]["metrics"]
            if mutation == "seed": row["host_seed"] += 1
            elif mutation == "seed_bool": row["host_seed"] = True
            elif mutation == "profile": row["profile"] = "monowheel"
            elif mutation == "kind": row["kind"] = "shared_graph"
            elif mutation == "phase": row["phase"] = "teacher"
            elif mutation == "task_hash": row["task_spec_sha256"] = "0" * 64
            elif mutation == "reference_hash": row["policy_sha256"] = "0" * 64
            elif mutation == "source_hash": row["boundary_audit_sha256"] = None
            elif mutation == "source": row["source_boundary_regressions"] = 1
            elif mutation == "error": row["error"] = {"code": "failed_fixture"}
            elif mutation == "owner": row["actual_owner_closed"] = False
            elif mutation == "cap": metrics["administratively_capped"] = True
            elif mutation == "integrity": metrics["integrity_failure"] = True
            elif mutation == "admission": metrics["failed_admissions"] = 1
            elif mutation == "success_bool": metrics["success"] = 1
            elif mutation == "owner_feel": metrics["owner_feel_accepted"] = True
            elif mutation == "latency": metrics["inference_p95_ms"] = 5.01
            elif mutation == "missing_latency": metrics["inference_p95_ms"] = None
            elif mutation == "nan": metrics["peak_useful_speed_cells_per_second"] = float("nan")
            elif mutation == "reversed_quantiles": metrics["inference_p50_ms"] = 4.
            elif mutation == "ticks": row["completed_ticks"] = 11
            elif mutation == "decisions_bool": row["decisions"] = True
            elif mutation == "task_ticks": metrics["elapsed_ticks"] = 6
            elif mutation == "success_time": metrics["success_seconds"] = .3
            elif mutation == "completion_time": metrics["capped_completion_seconds"] = 4.
            else: metrics["reason"] = "task_deadline"
            with self.subTest(mutation=mutation):
                self.assertFalse(evaluate_qualification(rows)["passed"])


if __name__ == "__main__":
    unittest.main()
