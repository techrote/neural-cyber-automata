"""Contract/controller fixtures only; actual game evidence is in test_skills_host."""

from copy import deepcopy
from dataclasses import replace
import unittest

from nca.contracts import ContractError
from nca.registry import Registry
from nca.skills import (ActorIdentity, SkillRuntime, UsableCapability,
                       interval_observations, register_skills, validate_parameters)
from nca.skills.usables import PROFILE_HASHES
from nca.skills.tasks import SkillSpec, TaskPlan, TaskStep, run_task


def capability(model="hammerhead"):
    parts, held, offset, usable = {
        "astra": (("chassis", "rotor_1", "rotor_2"), None, (3.5, 1), "launcher"),
        "hammerhead": (("chassis", "shoe_-1", "shoe_1", "hammer_arm", "hammer"), 4, (0, -1), "cutter"),
        "pouncer": (("chassis", "shoe_-1", "shoe_1", "glide_fin_-1", "glide_fin_1"), None, (4, 0), "depositor"),
    }[model]
    return UsableCapability(model, PROFILE_HASHES[model], parts, "usable", 0, held, offset, (-2.5, 2.5), usable)


def view(model="hammerhead", tick=0, *, generation=0, detached=False, epoch=1):
    cap = capability(model)
    identity = ActorIdentity("unit-run", 0, epoch, model, 1, generation)
    parts = [{"id": name, "index": i, "detached": detached if i == cap.held_part else False,
              "position": [204, 293] if i == cap.held_part else [200, 300], "rotation": 0}
             for i, name in enumerate(cap.part_ids)]
    return {**identity.to_dict(), "version": 2, "representation": "articulated-rapier",
            "observation_scope": "self, rig rays and bounded composed native cells",
            "observation_id": identity.observation_id(tick), "tick": tick,
            "position": [200, 300], "velocity": [0, 0], "rotation": 0, "angular_velocity": 0,
            "parts": parts, "joints": [], "tool_ready": True, "energy": 1, "charge": 0,
            "last_effects": [], "grounded": True, "suspended": False, "fault": ""}


class SkillContractTests(unittest.TestCase):
    def test_strict_skill_parameters_and_capability_hash(self):
        for skill, parameters in (("aim_use", {"aim": [float("nan"), 0]}),
                                  ("aim_use", {"tool": True}), ("grip", {"effort": True}),
                                  ("approach", {"target": [900, 400]}),
                                  ("release", {"timeout_ticks": 2}), ("invented", {})):
            with self.subTest(skill=skill, parameters=parameters), self.assertRaises(ContractError):
                validate_parameters(skill, parameters)
        with self.assertRaises(ContractError):
            replace(capability(), offset=(100, 0))
        with self.assertRaises(ContractError):
            SkillRuntime({"id": "hammerhead", "version": 1, "mounts": []})

    def test_initiation_is_owned_and_not_success_or_execution(self):
        runtime = SkillRuntime(capability("astra"))
        parameters = {"aim": [0, -1]}
        receipt = runtime.start("aim_use", parameters, view("astra"))
        parameters["aim"][0] = 1
        self.assertTrue(receipt.accepted)
        self.assertEqual(receipt.request.parameters["aim"], (0, -1))
        first = runtime.progress(view("astra"))
        self.assertFalse(first.action.tool)
        self.assertIsNone(first.local_success)
        self.assertIsNone(first.evaluator_success)

    def test_source_tick_is_never_restamped_and_expiry_is_exclusive(self):
        for age, accepted in ((119, True), (120, False)):
            runtime = SkillRuntime(capability("astra"))
            old = view("astra")
            receipt = runtime.start("aim_use", {}, view("astra", age),
                                    source_tick=0, source_observation_id=old["observation_id"])
            self.assertEqual(receipt.accepted, accepted)
            self.assertEqual(receipt.request.source_tick, 0)
            self.assertEqual(receipt.request.source_observation_id, old["observation_id"])

    def test_stale_mount_object_and_actor_refs_are_refused(self):
        runtime = SkillRuntime(capability())
        old = runtime.mount_ref(view())
        for stale, current in ((replace(old, object_part="replacement"), view()),
                               (old, view(detached=True)), (old, view(epoch=2))):
            receipt = runtime.start("grip", {}, current, mount_ref=stale)
            self.assertFalse(receipt.accepted)
            self.assertEqual(receipt.reason, "stale_mount_reference")

    def test_busy_resource_does_not_implicitly_take_over(self):
        runtime = SkillRuntime(capability())
        original = runtime.start("recover", {}, view(), request_id="original")
        other = runtime.start("release", {}, view(), source="human", source_generation=2)
        self.assertTrue(original.accepted)
        self.assertFalse(other.accepted)
        self.assertEqual(other.reason, "resource_busy")
        self.assertEqual(runtime.progress(view()).request_id, "original")

    def test_attachment_generation_survives_control_cancel_but_resets_with_object(self):
        runtime = SkillRuntime(capability())
        self.assertEqual(runtime.mount_ref(view()).attachment_generation, 0)
        detached = runtime.mount_ref(view(tick=6, detached=True))
        self.assertEqual(detached.attachment_generation, 1)
        cancelled_view = view(tick=6, generation=1, detached=True)
        cancelled = runtime.mount_ref(cancelled_view)
        self.assertEqual(cancelled.attachment_generation, 1)
        self.assertNotEqual(cancelled.actor, detached.actor)
        self.assertFalse(runtime.start("grip", {}, cancelled_view, mount_ref=detached).accepted)
        regripped = runtime.mount_ref(view(tick=12, generation=1))
        self.assertEqual(regripped.attachment_generation, 2)
        self.assertEqual(runtime.mount_ref(view(tick=12, generation=2)).attachment_generation, 2)
        self.assertEqual(runtime.mount_ref(view(epoch=2)).attachment_generation, 0)

    def test_invocation_id_is_idempotent_and_payload_collision_refused(self):
        runtime = SkillRuntime(capability())
        receipt = runtime.start("grip", {}, view(), request_id="once")
        done = runtime.progress(view())
        self.assertTrue(done.local_success)
        self.assertEqual(runtime.start("grip", {}, view(tick=12), request_id="once"), receipt)
        self.assertEqual(runtime.progress(view(tick=12)), done)
        with self.assertRaises(ContractError):
            runtime.start("release", {}, view(tick=12), request_id="once")

    def test_lifetime_change_invalidates_active_skill_before_trigger(self):
        runtime = SkillRuntime(capability("astra"))
        runtime.start("aim_use", {}, view("astra"))
        result = runtime.progress(view("astra", 6, generation=1))
        self.assertEqual(result.status, "invalidated")
        self.assertFalse(result.action.tool)
        self.assertTrue(result.cancel_host)

    def test_local_effect_feedback_proves_more_than_tool_request(self):
        runtime = SkillRuntime(capability("astra"))
        runtime.start("aim_use", {"require_change": True}, view("astra"))
        runtime.progress(view("astra"))
        self.assertTrue(runtime.progress(view("astra", 6)).action.tool)
        sample = view("astra", 7)
        sample["last_effects"] = [{"actor_slot": 0, "epoch": 1, "sequence": 1,
                                   "consumed": True, "status": "refused", "reason": "blocked",
                                   "changed_cells": 0, "attempted_writes": 0, "reads": 0}]
        result = runtime.progress(view("astra", 12), observations=(sample,))
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "effect_blocked")
        self.assertFalse(result.local_success)
        self.assertEqual(result.effects[0]["tick"], 7)

    def test_foreign_feedback_invalidates_and_gap_cannot_be_filled(self):
        runtime = SkillRuntime(capability("astra"))
        runtime.start("aim_use", {}, view("astra"))
        runtime.progress(view("astra", 6))
        result = runtime.progress(view("astra", 12), observations=(view("astra", 7, epoch=2),))
        self.assertEqual(result.status, "invalidated")
        with self.assertRaises(ContractError):
            interval_observations({"consumed_trace_complete": False, "trace": []})

    def test_same_mechanics_for_human_script_and_model_intent(self):
        actions = []
        for source in ("human", "script", "model"):
            runtime = SkillRuntime(capability("astra"))
            runtime.start("aim_use", {}, view("astra"), source=source)
            runtime.progress(view("astra"))
            actions.append(runtime.progress(view("astra", 6)).action)
        self.assertEqual(actions[0], actions[1])
        self.assertEqual(actions[1], actions[2])

    def test_recovery_never_uses_vectors_mode_button_as_self_right(self):
        for model, auxiliary in (("astra", False), ("hammerhead", True)):
            runtime = SkillRuntime(capability(model))
            current = view(model); current["rotation"] = 0.7
            runtime.start("recover", {}, current)
            self.assertEqual(runtime.progress(current).action.auxiliary, auxiliary)

    def test_no_progress_and_charge_cancel_require_owner_revocation(self):
        runtime = SkillRuntime(capability())
        runtime.start("approach", {"target": [250, 300], "no_progress_ticks": 6}, view())
        runtime.progress(view())
        failed = runtime.progress(view(tick=6))
        self.assertEqual(failed.reason, "no_progress")
        self.assertTrue(failed.cancel_host)
        charged = SkillRuntime(capability("pouncer"))
        charged.start("charge_hold", {}, view("pouncer"))
        self.assertTrue(charged.progress(view("pouncer")).action.primary)
        cancelled = charged.cancel("takeover")
        self.assertEqual(cancelled.status, "cancelled")
        self.assertTrue(cancelled.cancel_host)
        self.assertFalse(cancelled.action.primary)

    def test_registry_and_bounded_task_definition(self):
        registry = Registry(); register_skills(registry)
        instance = registry.create("skill", "authored-usables", profile=capability())
        self.assertIn("grip", instance.allowed_skills)
        with self.assertRaises(ContractError):
            TaskPlan("huge", (TaskStep(SkillSpec("recover")),) * 13)
        with self.assertRaises(ContractError):
            TaskStep(SkillSpec("recover"), (SkillSpec("recover"),) * 3)

    def test_completed_effect_time_wins_over_later_report_deadline(self):
        for event_tick, expected in ((7, "succeeded"), (13, "timed_out")):
            runtime = SkillRuntime(capability("astra"))
            runtime.start("aim_use", {"timeout_ticks": 12, "no_progress_ticks": 6}, view("astra"))
            runtime.progress(view("astra"))
            self.assertTrue(runtime.progress(view("astra", 6)).action.tool)
            sample = view("astra", event_tick)
            sample["last_effects"] = [{"actor_slot": 0, "epoch": 1, "sequence": 1,
                                       "consumed": True, "status": "applied", "reason": "",
                                       "changed_cells": 1, "attempted_writes": 1, "reads": 3}]
            result = runtime.progress(view("astra", max(12, event_tick)), observations=(sample,))
            self.assertEqual(result.status, expected)
            self.assertEqual(result.effects[0]["tick"], event_tick)

    def test_task_exception_disarms_charge_without_a_release_step(self):
        class FailedEnv:
            steps = 0
            cancellations = 0
            action = None

            def step(self, action):
                self.steps += 1
                self.action = action
                raise RuntimeError("original transport failure")

            def cancel(self):
                self.cancellations += 1
                raise OSError("cancellation connection is also unavailable")

        env = FailedEnv()
        runtime = SkillRuntime(capability("pouncer"))
        plan = TaskPlan("failed-owner", (TaskStep(SkillSpec("charge_hold")),))
        with self.assertRaisesRegex(RuntimeError, "original transport failure"):
            run_task(env, runtime, plan, view("pouncer"))
        self.assertEqual(env.steps, 1)
        self.assertTrue(env.action["primary"])
        self.assertEqual(env.cancellations, 1)
        self.assertEqual(runtime.progress(view("pouncer")).status, "cancelled")


if __name__ == "__main__":
    unittest.main()
