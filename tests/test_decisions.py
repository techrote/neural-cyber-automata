from dataclasses import replace
import threading
import time
import unittest

from nca.adapters.decisions import (CallableWorkerProvider, ControlIdentity,
    DecisionRouter, DelayedMockProvider, SkillProposal, decision_view, mock_proposal)
from nca.contracts import ContractError


def view(tick=0, run="run-a", epoch=1):
    return {"version": 2, "representation": "articulated-rapier",
            "observation_scope": "self, rig rays and bounded composed native cells",
            "run_id": run, "actor_id": 0, "epoch": epoch, "model": "jev",
            "profile_version": 1, "control_generation": 0, "observation_id": f"{run}-{tick}", "tick": tick,
            "position": [1.0, 2.0], "velocity": [0.0, 0.0],
            "evaluator": {"hidden_target": [999.0, 999.0]}, "reward": 400.0}


def validate(skill, parameters):
    if skill != "approach" or set(parameters) != {"distance"} or not 0 < parameters["distance"] <= 100:
        raise ValueError("invalid bounded parameters")


def provider(delay=0, **kwargs):
    return DelayedMockProvider(lambda request: mock_proposal(request, "approach", distance=10.0),
                               delay_ticks=delay, **kwargs)


def router(backend=None, **kwargs):
    events = []
    route = DecisionRouter(backend or provider(), ("approach",), validate, events.append, **kwargs)
    route.transfer("model", ControlIdentity.from_view(view()))
    return route, events


class DecisionTests(unittest.TestCase):
    def test_cancel_hook_failure_leaves_no_owner_and_retains_unacknowledged_slot(self):
        for operation in ("transfer", "revoke"):
            with self.subTest(operation=operation):
                backend = provider(6, cancellation_ack=False)
                route, _ = router(backend)
                route.tick(view())
                def failing_cancel(reason):
                    raise RuntimeError("injected owner transport failure")
                route.on_revoke = failing_cancel
                with self.assertRaisesRegex(RuntimeError, "owner transport"):
                    if operation == "transfer":
                        route.transfer("human", ControlIdentity.from_view(view()))
                    else:
                        route.revoke("human_takeover")
                self.assertIsNone(route.grant)
                self.assertIsNotNone(route._backend)
                self.assertTrue(route._retired)
                self.assertEqual(route.tick(view(6)).reason, "no_owner")
                self.assertEqual(route.accepted, 0)
                self.assertIsNone(route._backend)
                self.assertEqual(backend.calls, 1)

    def test_external_owner_cancel_fences_pending_reply_with_same_actor_epoch(self):
        route, canceled = router(provider(6, cancellation_ack=False))
        route.tick(view())
        latest = {**view(6), "control_generation": 1}
        result = route.tick(latest)
        self.assertEqual(result.reason, "lifetime_changed")
        self.assertEqual(canceled, ["lifetime_changed"])
        self.assertEqual(route.accepted, 0)
        self.assertIsNone(route.grant)
        self.assertEqual(route.tick(latest).reason, "no_owner")

    def test_transfer_uses_fresh_identity_returned_by_actual_cancel_hook(self):
        route, _ = router(provider(6))
        route.tick(view())
        fresh = ControlIdentity.from_view({**view(6), "control_generation": 1})
        route.on_revoke = lambda reason: fresh
        grant = route.transfer("human", ControlIdentity.from_view(view(6)))
        self.assertEqual(grant.identity.control_generation, 1)
        self.assertEqual(route.tick({**view(6), "control_generation": 1}).reason, "human_owns_actor")

    def test_malformed_provider_envelopes_fall_back_and_cannot_free_busy_slot(self):
        for reply in (("one",), (), {"bad": "envelope"}, ("x" * 300, {}), (7, {})):
            with self.subTest(reply=reply):
                backend = provider(outage=True, cancellation_ack=False)
                route, _ = router(backend)
                route.tick(view())
                backend.poll = lambda tick: reply
                self.assertEqual(route.tick(view(6)).reason, "provider_poll_failed")
                self.assertIsNotNone(route._backend)
                route.revoke()
                self.assertEqual(route.tick(view(7)).reason, "no_owner")
                self.assertIsNotNone(route._backend)
                self.assertEqual(backend.calls, 1)

    def test_nonboolean_cancel_acknowledgement_does_not_allow_second_call(self):
        backend = provider(outage=True, cancellation_ack=False)
        route, _ = router(backend)
        route.tick(view())
        backend.cancel = lambda request_id: {"acknowledged": False}
        self.assertTrue(route.tick(view(120)).backend_busy)
        self.assertTrue(route.tick(view(121)).backend_busy)
        self.assertEqual(backend.calls, 1)

    def test_delay_matrix_preserves_original_source_and_exclusive_expiry(self):
        for delay in (0, 6, 30, 120, 240):
            with self.subTest(delay=delay):
                route, _ = router(provider(delay))
                self.assertEqual(route.tick(view()).reason, "requested")
                result = route.tick(view(delay))
                if delay < 120:
                    self.assertEqual(result.status, "accepted")
                    self.assertEqual(result.proposal.observed_tick, 0)
                    self.assertEqual(result.proposal.observation_id, "run-a-0")
                    self.assertEqual(result.source_age_ticks, delay)
                else:
                    self.assertEqual(result.reason, "source_expired")
                    self.assertIsNone(result.proposal)
                self.assertTrue(result.local_control_required)

    def test_takeover_does_not_wait_for_hung_backend_or_renew_old_grant(self):
        backend = provider(240, cancellation_ack=False)
        route, canceled = router(backend)
        route.tick(view())
        previous = route.grant
        human = route.transfer("human", ControlIdentity.from_view(view(6)))
        self.assertEqual(canceled, ["takeover"])
        self.assertGreater(human.generation, previous.generation)
        self.assertEqual(route.tick(view(6)).reason, "retired_backend_busy")
        self.assertEqual(route.tick(view(240)).reason, "retired_reply")
        self.assertEqual(backend.calls, 1)
        self.assertEqual(route.tick(view(241)).reason, "human_owns_actor")

    def test_unacknowledged_timeout_cannot_accumulate_abandoned_calls(self):
        backend = provider(outage=True, cancellation_ack=False)
        route, _ = router(backend)
        route.tick(view())
        self.assertEqual(route.tick(view(120)).reason, "source_expired")
        for tick in range(121, 501):
            self.assertEqual(route.tick(view(tick)).reason, "retired_backend_busy")
        self.assertEqual(backend.calls, 1)
        self.assertEqual(backend.max_inflight, 1)

    def test_wall_timeout_expires_during_paused_simulation(self):
        clock = [0.0]
        route, _ = router(provider(outage=True), clock=lambda: clock[0], wall_timeout_seconds=2.0)
        route.tick(view())
        clock[0] = 2.0
        self.assertEqual(route.tick(view()).reason, "wall_timeout")

    def test_reset_and_mount_change_revoke_at_owner_boundary(self):
        for new_view, mount in ((view(0, "run-b", 2), 0), (view(6), 1)):
            route, canceled = router(provider(30))
            route.tick(view())
            result = route.tick(new_view, mount_generation=mount)
            self.assertEqual(result.reason, "lifetime_changed")
            self.assertEqual(canceled, ["lifetime_changed"])
            self.assertIsNone(route.grant)

    def test_late_reply_cannot_forge_fresh_observation(self):
        backend = DelayedMockProvider(lambda request: replace(mock_proposal(request, "approach", distance=5),
                                                        observed_tick=30, observation_id="run-a-30"),
                                      delay_ticks=30)
        route, _ = router(backend)
        route.tick(view())
        self.assertEqual(route.tick(view(30)).reason, "invalid_proposal")

    def test_no_perception_privilege_in_source_payload(self):
        source = view()
        copied = decision_view(source)
        self.assertNotIn("evaluator", copied)
        self.assertNotIn("reward", copied)
        source["position"][0] = 999.0
        self.assertEqual(copied["position"], [1.0, 2.0])
        source["observation_scope"] = "global evaluator"
        with self.assertRaises(ContractError):
            decision_view(source)

    def test_nonfinite_unknown_skills_and_executable_parameters_are_refused(self):
        choices = (
            lambda r: mock_proposal(r, "terrain_write", distance=1),
            lambda r: mock_proposal(r, "approach", distance=200),
            lambda r: mock_proposal(r, "approach", script="print('arbitrary code')"),
            lambda r: {**mock_proposal(r, "approach", distance=1).to_dict(), "parameters": {"distance": float("nan")}},
        )
        for choose in choices:
            route, _ = router(DelayedMockProvider(choose))
            route.tick(view())
            self.assertEqual(route.tick(view(1)).reason, "invalid_proposal")

    def test_reordering_cannot_clear_current_slot_and_duplicates_cannot_repeat_skill(self):
        class Injected:
            def __init__(self):
                self.request = None
                self.responses = []
            def submit(self, request, tick):
                self.request = request
            def poll(self, tick):
                return self.responses.pop(0) if self.responses else None
            def cancel(self, request_id):
                return True
        backend = Injected()
        route, _ = router(backend)
        route.tick(view())
        reply = (backend.request.id, mock_proposal(backend.request, "approach", distance=1))
        backend.responses = [("foreign-request", reply[1]), reply, reply]
        self.assertEqual(route.tick(view(1)).reason, "request_mismatch")
        self.assertEqual(route.tick(view(2)).status, "accepted")
        self.assertEqual(route.tick(view(3)).reason, "duplicate_or_unsolicited")
        self.assertEqual(route.accepted, 1)

    def test_request_cadence_is_measured_from_submission(self):
        backend = provider()
        route, _ = router(backend)
        route.tick(view())
        route.tick(view(1))
        self.assertEqual(route.tick(view(29)).reason, "request_cadence")
        self.assertEqual(route.tick(view(30)).reason, "requested")
        self.assertEqual(backend.calls, 2)

    def test_backend_worker_poll_remains_nonblocking_until_actual_completion(self):
        finished = threading.Event()
        def blocked(request):
            finished.wait(timeout=2.0)
            return mock_proposal(request, "approach", distance=3)
        backend = CallableWorkerProvider(blocked)
        route, _ = router(backend)
        try:
            route.tick(view())
            self.assertFalse(backend.cancel("anything"))
            route.tick(view(120))
            self.assertEqual(route.tick(view(121)).reason, "retired_backend_busy")
            finished.set()
            deadline = time.monotonic() + 1.0
            result = route.tick(view(122))
            while result.reason == "retired_backend_busy" and time.monotonic() < deadline:
                time.sleep(.001)
                result = route.tick(view(122))
            self.assertEqual(result.reason, "retired_reply")
        finally:
            finished.set()

    def test_source_cannot_supply_its_own_ownership_priority(self):
        route, _ = router()
        with self.assertRaises(ContractError):
            route.transfer("superuser", ControlIdentity.from_view(view()))
        self.assertEqual(route.grant.source, "model")

    def test_jitter_and_provider_faults_do_not_stop_local_control(self):
        backend = provider(6, jitter_ticks=(-2, 3))
        route, _ = router(backend)
        route.tick(view())
        self.assertEqual(route.tick(view(4)).status, "accepted")
        route.tick(view(30))
        self.assertEqual(route.tick(view(38)).reason, "awaiting_provider")
        self.assertEqual(route.tick(view(39)).status, "accepted")
        class Broken:
            def submit(self, request, tick):
                pass
            def poll(self, tick):
                raise RuntimeError("private backend details")
            def cancel(self, request_id):
                raise RuntimeError("private backend details")
        route, _ = router(Broken())
        result = route.tick(view())
        self.assertEqual(result.reason, "provider_poll_failed")
        self.assertTrue(result.local_control_required)
        self.assertNotIn("private", str(route.events))

    def test_unknown_host_profile_is_refused_before_provider_request(self):
        altered = view()
        altered["profile_version"] = 999
        with self.assertRaises(ContractError):
            ControlIdentity.from_view(altered)


if __name__ == "__main__":
    unittest.main()
