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
        class Env:
            def __init__(self): self.calls = []
            def cancel(self): self.calls.append("cancel"); return {"cancelled": True}
            def command(self, *args, **kwargs): self.calls.append("command"); raise AssertionError()
            def step_packet(self, packet): self.calls.append("step"); raise AssertionError()
        env = Env()
        fallback = ExportController(bundle()).cancel("profile_mismatch")
        self.assertEqual(dispatch(env, fallback, view()), {"cancelled": True})
        self.assertEqual(env.calls, ["cancel"])

    def test_dispatch_preserves_original_packet_and_cleans_up_transport_error(self):
        class Env:
            def __init__(self): self.original = None; self.cancelled = 0
            def command(self, action, *, observation): self.original = observation; return {"request": action}
            def step_packet(self, packet): raise RuntimeError("transport failure")
            def cancel(self): self.cancelled += 1; raise OSError("cleanup failure")
        env = Env()
        observation = view()
        decision = ExportController(bundle()).step(observation, self.goal)
        with self.assertRaisesRegex(RuntimeError, "transport failure"):
            dispatch(env, decision, observation)
        self.assertIs(env.original, observation)
        self.assertEqual(env.cancelled, 1)

    def test_dispatch_command_construction_failure_cancels_and_preserves_error(self):
        class Env:
            def __init__(self): self.calls = []
            def command(self, *args, **kwargs):
                self.calls.append("command")
                raise RuntimeError("source invalidated")
            def step_packet(self, packet): self.calls.append("step"); raise AssertionError()
            def cancel(self): self.calls.append("cancel"); raise OSError("cleanup failed")
        env = Env()
        observation = view()
        decision = ExportController(bundle()).step(observation, self.goal)
        with self.assertRaisesRegex(RuntimeError, "source invalidated"):
            dispatch(env, decision, observation)
        self.assertEqual(env.calls, ["command", "cancel"])

    def test_dispatch_cannot_refresh_old_inference_with_newer_or_changed_view(self):
        class Env:
            def __init__(self): self.calls = []
            def command(self, *args, **kwargs): self.calls.append("command"); raise AssertionError()
            def step_packet(self, packet): self.calls.append("step"); raise AssertionError()
            def cancel(self): self.calls.append("cancel")
        for changed in (view(6), view(control=1), view(epoch=2), dict(view(), position=[177., 346.5])):
            env = Env()
            original = view()
            decision = ExportController(bundle()).step(original, self.goal)
            with self.assertRaisesRegex(ContractError, "decision_source_mismatch"):
                dispatch(env, decision, changed)
            self.assertEqual(env.calls, ["cancel"])


if __name__ == "__main__":
    unittest.main()
