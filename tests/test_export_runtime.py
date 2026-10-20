"""Numeric/contract fixtures; these do not qualify a trained game deployment."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_NUMPY:
    import numpy as np
    from nca.contracts import ContractError, WireAction, sha256_json
    from nca.controllers.baseline import SkillGoal
    from nca.export import ExportController, RuntimeBundle, NumpyGRU, context_for, dispatch
    from nca.export.build import immutable_json, make_bundle
    from nca.export.runtime import encode, map_action
    from nca.export.spec import EXPECTED_SPECIFICATION, WEIGHT_SHAPES, PINNED_HOST
    from nca.adapters.protocol import make_physical_command

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    profile = json.loads((ROOT / "configs/export/jev-profile-v1.json").read_text())
    context = context_for(profile, PINNED_HOST)
    # Portable contract fixture describing the one candidate Linux context.
    # It is not evidence that the platform running this unit test is qualified.
    context = replace(context, python="3.12.14", numpy="2.3.5", platform="Linux", architecture="x86_64")
    rng = np.random.default_rng(73109)
    numeric = {"format": "nca-gru-numeric/v1", "specification": EXPECTED_SPECIFICATION,
               "matrix_layout": "row_major_out_features_by_in_features",
               "gru_gate_order": ["reset", "update", "new"],
               "gru_equations": {"r": "sigmoid(W_ir*x+b_ir+W_hr*h+b_hr)",
                                 "z": "sigmoid(W_iz*x+b_iz+W_hz*h+b_hz)",
                                 "n": "tanh(W_in*x+b_in+r*(W_hn*h+b_hn))", "h_next": "(1-z)*n+z*h"},
               "weights": {name: rng.normal(0, .07, shape).astype(np.float32).tolist()
                           for name, shape in WEIGHT_SHAPES.items()}}
    numeric["policy_sha256"] = NumpyGRU(numeric["weights"]).policy_sha256
    # Deliberately synthetic in-memory evidence only, never a deployment artifact.
    evidence = {key: "a"*64 for key in ("checkpoint_sha256", "numeric_source_sha256", "summary_sha256",
                "config_sha256", "manifest_sha256", "learner_sources_sha256", "dataset_sha256")}
    evidence.update(status="trained_and_evaluated", kind="bc", training_seed=1103,
                    training_privilege="none", profile_sha256=context.profile_sha256)
    return numeric, profile, context, evidence


def bundle():
    numeric, profile, context, evidence = fixture()
    return RuntimeBundle(make_bundle(numeric, profile, context, evidence), context)


def view(tick=0, epoch=1, control=0):
    return {"version": 2, "representation": "articulated-rapier",
            "observation_scope": "self, rig rays and bounded composed native cells",
            "run_id": "export-test", "actor_id": 0, "epoch": epoch, "model": "jev", "profile_version": 1,
            "control_generation": control, "tick": tick,
            "observation_id": f"export-test:{epoch}:0:{control}:{tick}",
            "position": [176., 346.5], "velocity": [2., -3.], "rotation": .12,
            "angular_velocity": -.4, "grounded": True, "energy": .8, "cooldown": .2,
            "ground_distance": 4., "ceiling_distance": 96., "fault": "", "suspended": False,
            "parts": [{"id": name, "index": i, "detached": False}
                      for i, name in enumerate(("impact_core", "rebound_pad_-1", "rebound_pad_1"))]}


class NativeReceiptFixture:
    """Synthetic native-boundary records; no physics or deployment evidence."""
    def __init__(self, observation=None):
        self.view = deepcopy(view() if observation is None else observation)
        self.calls = []
        self.packets = []
        self.sequence = -1
        self.fail_command = self.fail_step = self.fail_cancel = False
        self.cancel_generation_delta = 1
        self.mutate_receipt = None
        self.terminal_tick = None

    def command(self, action, *, observation, lease_ticks):
        self.calls.append("command")
        if self.fail_command:
            raise RuntimeError("source invalidated")
        self.sequence += 1
        return make_physical_command(observation, self.sequence, action, lease_ticks)

    def step_packet(self, packet, *, ticks):
        self.calls.append("step")
        self.packets.append(deepcopy(packet))
        if self.fail_step:
            raise RuntimeError("transport failure")
        start = self.view["tick"]
        count = min(ticks, self.terminal_tick - start) if self.terminal_tick is not None else ticks
        traces = []
        for offset in range(count):
            terminal = self.terminal_tick == start + offset + 1
            following = deepcopy(self.view)
            following.update(tick=start + offset + 1, suspended=terminal,
                             control_generation=self.view["control_generation"] + int(terminal))
            following["observation_id"] = ":".join(str(following[key]) for key in
                ("run_id", "epoch", "actor_id", "control_generation", "tick"))
            traces.append({"tick_before": start + offset, "tick_after": start + offset + 1,
                "failed": False, "requested": [deepcopy(packet)] if offset == 0 else [],
                "admission": [{"accepted": True, "reason": ""}] if offset == 0 else [],
                "observation": following,
                "applied": {"action_available": not terminal, "terminal_revocation": terminal,
                            "action": None if terminal else deepcopy(packet["action"])}})
        self.view = deepcopy(traces[-1]["observation"])
        info = {"actual_ticks": count, "trace": traces, "idle_trace": [], "idle_trace_dropped": 0,
                "consumed_trace_complete": True, "unconsumed_terminal_commands": []}
        result = (deepcopy(self.view), 0., False, self.view["suspended"], info)
        if self.mutate_receipt is not None:
            self.mutate_receipt(result)
        return result

    def cancel(self):
        self.calls.append("cancel")
        if self.fail_cancel:
            raise OSError("cleanup failed")
        self.view["control_generation"] += self.cancel_generation_delta
        self.view["observation_id"] = ":".join(str(self.view[key]) for key in
            ("run_id", "epoch", "actor_id", "control_generation", "tick"))
        return deepcopy(self.view)


@unittest.skipUnless(HAS_NUMPY, "optional NumPy CPU export dependency unavailable")
class ExportTests(unittest.TestCase):
    def setUp(self):
        self.goal = SkillGoal("hop", "directed_hop", (200, 346.5), timeout_seconds=4)

    @unittest.skipUnless(HAS_TORCH, "optional Torch float reference unavailable")
    def test_independent_sequential_fp32_gru_matches_torch_including_reset(self):
        import torch
        torch.set_num_threads(1)
        numeric, *_ = fixture()
        network = NumpyGRU(numeric["weights"])
        cell = torch.nn.GRUCell(32, 32)
        for name, value in cell.state_dict().items():
            value.copy_(torch.tensor(numeric["weights"]["recurrent."+name]))
        w = {name: torch.tensor(value) for name, value in numeric["weights"].items()}
        rng = np.random.default_rng(73109)
        nh = np.zeros(32, dtype=np.float32)
        th = torch.zeros(32)
        with torch.no_grad():
            for index in range(128):
                x = rng.uniform(-2, 2, 26).astype(np.float32)
                reset = index % 32 == 0
                if reset:
                    th.zero_()
                result, value, nh, logits = network.forward(x, nh, reset=reset)
                th = cell(torch.tanh(w["encoder.weight"] @ torch.tensor(x)+w["encoder.bias"]), th)
                expected_logits = w["actor.weight"] @ th+w["actor.bias"]
                expected_value = float((w["critic.weight"] @ th+w["critic.bias"])[0])
                expected = torch.cat((expected_logits[:2].tanh(), expected_logits[2:].sigmoid()))
                np.testing.assert_allclose(nh, th.numpy(), rtol=0, atol=2e-5)
                np.testing.assert_allclose(logits, expected_logits.numpy(), rtol=0, atol=2e-5)
                np.testing.assert_allclose(result, expected.numpy(), rtol=0, atol=2e-5)
                self.assertLessEqual(abs(value-expected_value), 2e-5)

    def test_exact_preprocessing_masks_history_and_unit_aim(self):
        observation = view()
        observation.update(grounded=None, energy=None, cooldown=None, ground_distance=None, ceiling_distance=None)
        prior = WireAction(move=(.25, 0), trim=-.5, effort=.7, primary=True)
        values = encode(observation, self.goal, prior, 0)
        self.assertEqual(len(values), 26)
        self.assertEqual(values[7:17], (0.,)*10)
        self.assertEqual(values[-6:], (.25, -.5, .7, 1., 0., 0.))
        self.assertEqual(values[0], 24/64)
        action = map_action([0, 0, .5, .5, .499, .5], observation, self.goal)
        self.assertAlmostEqual(sum(v*v for v in action.aim), 1)
        self.assertTrue(action.primary and action.auxiliary)
        self.assertFalse(action.secondary or action.tool or action.attach_usable)

    def test_shape_nonfinite_and_unknown_weight_refusal(self):
        numeric, *_ = fixture()
        for change in (lambda w: w.update(extra=[]), lambda w: w.update({"actor.bias": [0]*5}),
                       lambda w: w["encoder.bias"].__setitem__(0, float("nan")),
                       lambda w: w.update({"log_std": [True]*3}),
                       lambda w: w.update({"encoder.weight": [[0], [0, 1]]})):
            weights = deepcopy(numeric["weights"])
            change(weights)
            with self.assertRaises(ContractError):
                NumpyGRU(weights)

    def test_altered_gate_order_and_candidate_equations_are_refused(self):
        for field, wrong in (("gru_gate_order", ["update", "reset", "new"]), ("gru_equations", {})):
            numeric, profile, context, evidence = fixture()
            numeric[field] = wrong
            with self.assertRaises(ContractError):
                make_bundle(numeric, profile, context, evidence)

    def test_bundle_owns_metadata_and_weights(self):
        item = bundle()
        exposed = item.value
        exposed["profile"]["parts"][0]["id"] = "other"
        exposed["weights"]["actor.bias"][0] = 999
        self.assertEqual(item.part_ids[0], "impact_core")
        self.assertNotEqual(item.network.weights["actor.bias"][0], 999)
        with self.assertRaises(TypeError):
            item.network.weights["actor.bias"] = np.zeros(6)

    def test_context_rejects_timing_profile_sensor_action_assist_and_platform_changes(self):
        item = bundle()
        changes = {"profile_sha256": "b"*64, "sensor_sha256": "b"*64, "action_sha256": "b"*64,
                   "assist_sha256": "b"*64, "normalization_sha256": "b"*64, "base_hz": 30,
                   "decision_ticks": 12, "helper_lease_ticks": 18, "platform": "Windows"}
        for name, wrong in changes.items():
            with self.subTest(name=name), self.assertRaises(ContractError):
                RuntimeBundle(item.value, replace(item.context, **{name: wrong}))

    def test_version_normalization_source_and_missing_training_evidence_rejected(self):
        item = bundle()
        for field, wrong in (("format", "nca-cpu-policy/v2"), ("normalization", {}),
                             ("runtime_sources", {}), ("training_evaluation", {})):
            changed = item.value
            changed[field] = wrong
            changed.pop("sha256")
            changed["sha256"] = sha256_json(changed)
            with self.assertRaises(ContractError):
                RuntimeBundle(changed, item.context)

    def test_immutable_file_hash_is_required_before_loading(self):
        item = bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"policy.json"
            digest = immutable_json(path, item.value)
            self.assertEqual(RuntimeBundle.load(path, item.context, expected_sha256=digest).sha256, item.sha256)
            with self.assertRaises(ContractError):
                RuntimeBundle.load(path, item.context, expected_sha256="0"*64)
            with self.assertRaises(ContractError):
                immutable_json(path, item.value)

    def test_cancellation_and_source_change_require_fresh_goal(self):
        for change in ({"source_generation": 2}, {"control": 1}, {"mount": 1}):
            controller = ExportController(bundle())
            first = controller.step(view(), self.goal)
            self.assertEqual(first.status, "accepted")
            later = view(6, control=change.get("control", 0))
            if "mount" in change:
                later["mount_generation"] = change["mount"]
            result = controller.step(later, self.goal, source_generation=change.get("source_generation", 1))
            self.assertTrue(result.cancel_owner)
            self.assertEqual(result.action, WireAction(effort=0))
            self.assertTrue(np.equal(controller.hidden, 0).all())
        controller = ExportController(bundle())
        controller.step(view(), self.goal)
        controller.cancel()
        self.assertEqual(controller.step(view(6), self.goal).reason, "retired_goal")

    def test_actor_and_goal_recurrence_resets_do_not_resume_old_incarnation(self):
        controller = ExportController(bundle())
        controller.step(view(), self.goal)
        fresh = controller.step(view(0, epoch=2), self.goal)
        self.assertEqual(fresh.hidden_before, (0.,)*32)
        self.assertEqual(controller.step(view(6), self.goal).reason, "retired_actor")
        controller = ExportController(bundle())
        controller.step(view(), self.goal)
        fresh = controller.step(view(6), replace(self.goal, id="next"))
        self.assertEqual(fresh.hidden_before, (0.,)*32)
        self.assertEqual(fresh.features[-6:], (0.,)*6)

    def test_fresh_goal_after_delayed_cancel_resets_cadence_but_old_goal_stays_retired(self):
        controller = ExportController(bundle())
        controller.step(view(), self.goal)
        self.assertEqual(controller.step(view(), self.goal, current_tick=18).reason, "source_expired")
        self.assertEqual(controller.step(view(24, control=1), self.goal).reason, "retired_goal")
        fresh = controller.step(view(24, control=1), replace(self.goal, id="fresh-after-cancel"))
        self.assertEqual(fresh.status, "accepted")
        self.assertEqual(fresh.hidden_before, (0.,)*32)
        self.assertEqual(fresh.features[-6:], (0.,)*6)
        changed = controller.step(view(31, control=1), replace(self.goal, id="fresh-after-cancel"), source_generation=2)
        self.assertEqual(changed.reason, "owner_or_mount_changed")

    def test_fresh_goal_cannot_roll_authority_or_source_clock_backward_after_cancel(self):
        cases = ((view(18, control=1), 2, "retired_control_generation"),
                 (view(18, control=2), 1, "retired_source_generation"),
                 (view(6, control=2), 2, "source_reordered"))
        for later, source, reason in cases:
            controller = ExportController(bundle())
            accepted = controller.step(view(12, control=2), self.goal, source_generation=2)
            self.assertEqual(accepted.status, "accepted")
            controller.cancel()
            result = controller.step(later, replace(self.goal, id="fresh-id"), source_generation=source)
            self.assertEqual(result.reason, reason)
            self.assertTrue(result.cancel_owner)
            fresh = controller.step(view(0, epoch=2), replace(self.goal, id="new-actor"), source_generation=1)
            self.assertEqual(fresh.status, "accepted")
            self.assertEqual(fresh.hidden_before, (0.,)*32)

    def test_duplicate_gap_old_lease_and_forged_view_have_neutral_cancel_fallback(self):
        for later, clock in ((view(), None), (view(12), None), (view(6), 18)):
            controller = ExportController(bundle())
            controller.step(view(), self.goal)
            result = controller.step(later, self.goal, current_tick=clock)
            self.assertEqual(result.status, "fallback")
            self.assertTrue(result.cancel_owner)
        controller = ExportController(bundle())
        bad = view()
        bad["observation_id"] = "forged"
        self.assertEqual(controller.step(bad, self.goal).reason, "observation_identity")

    def test_dispatch_cancels_fallback_without_sending_neutral_release(self):
        env = NativeReceiptFixture()
        controller = ExportController(bundle())
        fallback = controller.cancel("profile_mismatch")
        receipt = dispatch(env, controller, fallback, view())
        self.assertEqual(receipt["control_generation"], 1)
        self.assertEqual(env.calls, ["cancel"])
        self.assertFalse(controller.owner_cleanup_unconfirmed)

    def test_dispatch_preserves_original_packet_and_cleans_up_transport_error(self):
        env = NativeReceiptFixture()
        env.fail_step = env.fail_cancel = True
        controller = ExportController(bundle())
        observation = view()
        decision = controller.step(observation, self.goal)
        with self.assertRaisesRegex(RuntimeError, "transport failure"):
            dispatch(env, controller, decision, observation)
        self.assertEqual(env.packets[0]["observation_id"], observation["observation_id"])
        self.assertEqual(env.calls, ["command", "step", "cancel"])
        self.assertIsNone(controller.pending)
        self.assertTrue(controller.owner_cleanup_unconfirmed)
        fresh = controller.step(view(6), replace(self.goal, id="after-error"))
        self.assertEqual(fresh.reason, "owner_cleanup_unconfirmed")
        controller.reset()
        self.assertTrue(controller.owner_cleanup_unconfirmed)
        env.fail_cancel = False
        controller.confirm_owner_cancel(env.cancel())
        self.assertFalse(controller.owner_cleanup_unconfirmed)
        fresh = controller.step(env.view, replace(self.goal, id="confirmed-fresh"))
        self.assertEqual(fresh.status, "accepted")
        self.assertEqual(fresh.hidden_before, (0.,) * 32)

    def test_dispatch_command_construction_failure_cancels_and_preserves_error(self):
        env = NativeReceiptFixture()
        env.fail_command = env.fail_cancel = True
        controller = ExportController(bundle())
        observation = view()
        decision = controller.step(observation, self.goal)
        with self.assertRaisesRegex(RuntimeError, "source invalidated"):
            dispatch(env, controller, decision, observation)
        self.assertEqual(env.calls, ["command", "cancel"])
        self.assertTrue(controller.owner_cleanup_unconfirmed)

    def test_dispatch_cannot_refresh_old_inference_with_newer_or_changed_view(self):
        for changed in (view(6), view(control=1), view(epoch=2), dict(view(), position=[177., 346.5])):
            env = NativeReceiptFixture()
            original = view()
            controller = ExportController(bundle())
            decision = controller.step(original, self.goal)
            with self.assertRaisesRegex(ContractError, "decision_source_mismatch"):
                dispatch(env, controller, decision, changed)
            self.assertEqual(env.calls, ["cancel"])

    def test_retired_replaced_cross_controller_and_copied_decisions_cannot_dispatch(self):
        for change in ("cancel", "reset", "new_goal", "new_source", "copy", "foreign_controller"):
            with self.subTest(change=change):
                env = NativeReceiptFixture()
                controller = ExportController(bundle())
                decision = controller.step(view(), self.goal)
                if change == "cancel":
                    controller.cancel()
                elif change == "reset":
                    controller.reset()
                elif change == "new_goal":
                    controller.step(view(6), replace(self.goal, id="replacement"))
                elif change == "new_source":
                    controller.step(view(6), replace(self.goal, id="replacement"), source_generation=2)
                elif change == "copy":
                    decision = replace(decision)
                else:
                    controller = ExportController(bundle())
                with self.assertRaisesRegex(ContractError, "decision_source_mismatch"):
                    dispatch(env, controller, decision, view())
                self.assertEqual(env.calls, ["cancel"])
                self.assertEqual(env.packets, [])

    def test_successful_decision_is_single_use_and_live_goal_change_requires_native_cancel(self):
        env = NativeReceiptFixture()
        controller = ExportController(bundle())
        decision = controller.step(view(), self.goal)
        following, *_ = dispatch(env, controller, decision, view())
        self.assertIsNone(controller.pending)
        replacement = controller.step(following, replace(self.goal, id="new"), source_generation=2)
        self.assertTrue(replacement.cancel_owner)
        self.assertEqual(replacement.reason, "owner_or_mount_changed")
        with self.assertRaisesRegex(ContractError, "decision_source_mismatch"):
            dispatch(env, controller, decision, view())
        self.assertEqual(len(env.packets), 1)
        self.assertFalse(controller.owner_cleanup_unconfirmed)
        current = env.view
        accepted = controller.step(current, replace(self.goal, id="after-revocation"), source_generation=2)
        self.assertEqual(accepted.status, "accepted")
        self.assertEqual(accepted.hidden_before, (0.,) * 32)

    def test_cancel_receipt_must_revoke_original_owner_and_not_rewind_clock(self):
        for mutate in (lambda v: v.update(control_generation=0),
                       lambda v: v.update(tick=0), lambda v: v.update(epoch=2),
                       lambda v: v.update(observation_id="forged"),
                       lambda v: v.pop("parts")):
            env = NativeReceiptFixture()
            controller = ExportController(bundle())
            decision = controller.step(view(), self.goal)
            dispatch(env, controller, decision, view())
            controller.cancel()
            receipt = env.cancel()
            mutate(receipt)
            if receipt.get("observation_id") != "forged":
                receipt["observation_id"] = ":".join(str(receipt[key]) for key in
                    ("run_id", "epoch", "actor_id", "control_generation", "tick"))
            with self.assertRaises(ContractError):
                controller.confirm_owner_cancel(receipt)
            self.assertTrue(controller.owner_cleanup_unconfirmed)
            self.assertEqual(controller.step(env.view, replace(self.goal, id="retry")).reason,
                             "owner_cleanup_unconfirmed")

    def test_old_or_same_epoch_reset_shape_cannot_clear_uncertain_owner(self):
        env = NativeReceiptFixture(view(epoch=5))
        env.fail_step = env.fail_cancel = True
        controller = ExportController(bundle())
        decision = controller.step(env.view, self.goal)
        with self.assertRaisesRegex(RuntimeError, "transport failure"):
            dispatch(env, controller, decision, env.view)
        for epoch in (3, 5):
            result = controller.step(view(epoch=epoch), replace(self.goal, id=f"old-{epoch}"))
            self.assertTrue(result.cancel_owner)
            self.assertTrue(controller.owner_cleanup_unconfirmed)
        fresh = controller.step(view(epoch=6), replace(self.goal, id="real-reset"))
        self.assertEqual(fresh.status, "accepted")
        self.assertFalse(controller.owner_cleanup_unconfirmed)
        self.assertEqual(fresh.hidden_before, (0.,) * 32)
        self.assertEqual(controller.step(view(epoch=3), replace(self.goal, id="unseen-old")).reason, "retired_actor")

    def test_goal_deadline_uses_current_tick_and_caps_original_lease(self):
        goal = replace(self.goal, timeout_seconds=.1)
        controller = ExportController(bundle())
        expired = controller.step(view(), goal, current_tick=6)
        self.assertTrue(expired.cancel_owner)
        self.assertEqual(expired.reason, "goal_deadline")
        controller = ExportController(bundle())
        accepted = controller.step(view(), goal, current_tick=5)
        self.assertEqual((accepted.helper_lease_ticks, accepted.goal_deadline_tick, accepted.checked_tick), (6, 6, 5))
        env = NativeReceiptFixture(view(5))
        following, _, _, _, info = dispatch(env, controller, accepted, view(), current_tick=5)
        self.assertEqual(info["actual_ticks"], 1)
        self.assertEqual(following["tick"], 6)
        self.assertEqual(env.packets[0]["observed_tick"], 0)
        self.assertEqual(env.packets[0]["lease_ticks"], 6)
        self.assertTrue(controller.owner_cleanup_unconfirmed)
        self.assertIsNone(controller.pending)

    def test_dispatch_checks_deadline_again_and_shortens_partial_final_interval(self):
        for now in (-1, 6, 12):
            controller = ExportController(bundle())
            goal = replace(self.goal, timeout_seconds=.1)
            decision = controller.step(view(), goal)
            env = NativeReceiptFixture()
            with self.assertRaisesRegex(ContractError, "dispatch_deadline"):
                dispatch(env, controller, decision, view(), current_tick=now)
            self.assertEqual(env.calls, ["cancel"])
        controller = ExportController(bundle())
        goal = replace(self.goal, timeout_seconds=7 / 60)
        env = NativeReceiptFixture()
        first = controller.step(view(), goal)
        following, *_ = dispatch(env, controller, first, view())
        second = controller.step(following, goal)
        self.assertEqual(second.helper_lease_ticks, 1)
        final, _, _, _, info = dispatch(env, controller, second, following)
        self.assertEqual((final["tick"], info["actual_ticks"]), (7, 1))
        self.assertEqual(env.packets[-1]["observed_tick"] + env.packets[-1]["lease_ticks"], 7)

    def test_delayed_original_view_can_dispatch_with_complete_live_receipts(self):
        controller = ExportController(bundle())
        decision = controller.step(view(), self.goal, current_tick=3)
        env = NativeReceiptFixture(view(3))
        following, *_ = dispatch(env, controller, decision, view())
        self.assertEqual(following["tick"], 9)
        self.assertEqual(env.packets[0]["observed_tick"], 0)
        self.assertFalse(controller.owner_cleanup_unconfirmed)

    def test_matching_actions_cannot_prove_consumption_after_original_lease_expiry(self):
        controller = ExportController(bundle())
        decision = controller.step(view(), self.goal, current_tick=9)
        env = NativeReceiptFixture(view(9))
        with self.assertRaisesRegex(ContractError, "dispatch_deadline"):
            dispatch(env, controller, decision, view())
        self.assertEqual(env.calls, ["cancel"])
        # A further native delay can occur after a valid dispatch-time check.
        # Even matching action values and unchanged synthetic owner metadata
        # cannot turn the expired suffix into confirmed consumption.
        controller = ExportController(bundle())
        decision = controller.step(view(), self.goal, current_tick=3)
        env = NativeReceiptFixture(view(9))
        with self.assertRaisesRegex(ContractError, "dispatch_receipt"):
            dispatch(env, controller, decision, view())
        self.assertEqual(env.calls, ["command", "step", "cancel"])
        self.assertEqual(tuple(controller.hidden), (0.,) * 32)

    def test_refused_mixed_missing_or_changed_receipts_cancel_and_cannot_retry(self):
        mutations = (
            lambda r: r[4]["trace"][0]["admission"][0].update(accepted=False),
            lambda r: r[4]["trace"][1].update(requested=[r[4]["trace"][0]["requested"][0]]),
            lambda r: r[4].update(consumed_trace_complete=False),
            lambda r: r[4]["trace"].pop(),
            lambda r: r[4]["trace"][2]["applied"]["action"].update(effort=0.),
            lambda r: r[0].update(mount_generation=1),
            lambda r: r[0]["parts"][1].update(detached=True),
        )
        for mutate in mutations:
            env = NativeReceiptFixture()
            env.mutate_receipt = mutate
            controller = ExportController(bundle())
            decision = controller.step(view(), self.goal)
            with self.assertRaises(ContractError):
                dispatch(env, controller, decision, view())
            self.assertEqual(env.calls, ["command", "step", "cancel"])
            self.assertIsNone(controller.pending)
            self.assertEqual(tuple(controller.hidden), (0.,) * 32)

    def test_echoed_packet_allows_only_json_rounding_with_exact_source_and_flags(self):
        for difference, accepted in ((1e-14, True), (1e-7, False)):
            env = NativeReceiptFixture()
            def mutate(result):
                action = result[4]["trace"][0]["requested"][0]["action"]
                action["effort"] += difference
            env.mutate_receipt = mutate
            controller = ExportController(bundle())
            decision = controller.step(view(), self.goal)
            if accepted:
                dispatch(env, controller, decision, view())
            else:
                with self.assertRaisesRegex(ContractError, "dispatch_refused"):
                    dispatch(env, controller, decision, view())

    def test_terminal_receipt_keeps_unavailable_last_action_and_retires_local_goal(self):
        env = NativeReceiptFixture()
        env.terminal_tick = 4
        controller = ExportController(bundle())
        decision = controller.step(view(), self.goal)
        final, _, terminated, truncated, info = dispatch(env, controller, decision, view())
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["actual_ticks"], 4)
        self.assertIsNone(info["trace"][-1]["applied"]["action"])
        self.assertTrue(controller.owner_cleanup_unconfirmed)
        self.assertEqual(controller.step(final, replace(self.goal, id="cannot-revive-terminal")).reason,
                         "actor_unavailable")


if __name__ == "__main__":
    unittest.main()
