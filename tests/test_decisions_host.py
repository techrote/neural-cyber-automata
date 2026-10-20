"""Stepped boundary and realtime native/Rapier conformance with mock providers."""

import os
import time
import unittest

from nca.adapters.decisions import DelayedMockProvider, mock_proposal
from nca.adapters.skill_control import SkillControlSession
from nca.contracts import ContractError
from nca.envs import CybersandEnv, GodotHost, physical_config


class _DecisionCases:
    @classmethod
    def setUpClass(cls):
        cls.host = GodotHost(os.environ["NCA_HOST_CHECKOUT"], os.environ["NCA_GODOT"], mode=cls.mode)

    @classmethod
    def tearDownClass(cls):
        cls.host.close()

    def session(self, delay=0, *, model="astra", ticks=480, **kwargs):
        host = self.host
        config = physical_config(model, ticks=ticks, seed=73)
        if model != "pouncer":
            config["actors"][0]["position"] = [300, 300]
        env = CybersandEnv(host, config)
        view, _ = env.reset()
        profile = host.request("profile")["profile"]
        backend = DelayedMockProvider(lambda req: mock_proposal(req, "aim_use", aim=[0, -1]),
                                      delay_ticks=delay, **kwargs)
        session = SkillControlSession(env, profile, view, backend)
        session.transfer("model")
        return session, backend

    @staticmethod
    def native_expiries(results, cancellations):
        """Find revocations proved by consecutive owner receipts and the admitted lease.

        A slow controller can miss the twelve-tick native lease even when the
        provider's 120-tick plan is still valid. An arbitrary lost grant, missing
        feedback, or a test's explicit cancel is not evidence of that expiry.
        """
        traces = {}
        for result in results:
            for trace in (*result.info.get("idle_trace", ()), *result.info["trace"]):
                traces[trace["tick_after"]] = trace
        previous = None
        deadline = None
        expired = []
        explicit_generations = {receipt["control_generation"] for receipt in cancellations}
        for tick, trace in sorted(traces.items()):
            admitted = [packet for packet, receipt in zip(trace["requested"], trace["admission"])
                        if receipt.get("accepted") is True]
            if admitted:
                deadline = admitted[-1]["observed_tick"] + admitted[-1]["lease_ticks"]
            view = trace["observation"]
            if (previous is not None and tick == previous["tick"] + 1
                    and view["control_generation"] == previous["control_generation"] + 1
                    and view["control_generation"] not in explicit_generations
                    and deadline is not None and trace["tick_before"] == deadline
                    and not admitted and previous["lease_remaining"] == 0
                    and view["lease_remaining"] == 0 and view["charge"] == 0
                    and not view["suspended"] and not view["fault"]):
                expired.append(view["control_generation"])
            previous = view
        return expired


@unittest.skipUnless(os.environ.get("NCA_HOST_CHECKOUT") and os.environ.get("NCA_GODOT"),
                     "actual owner is not configured; no asynchronous game qualification")
class SteppedDecisionTests(_DecisionCases, unittest.TestCase):
    """Exact feedback/proposal contracts with actual physics and a paused owner boundary."""

    mode = "stepped"

    def test_stepped_delay_matrix_preserves_source_expiry_and_drains_old_reply(self):
        # Exact router-boundary expectations use actual stepped physics. The
        # realtime case below additionally exercises independent native expiry.
        for delay in (0, 6, 30, 120, 240):
            with self.subTest(delay=delay):
                session, backend = self.session(delay, cancellation_ack=False)
                updates = []
                for _ in range(44):
                    result = session.step()
                    updates.append(result.decision)
                    self.assertTrue(result.info["consumed_trace_complete"])
                    self.assertEqual(result.info["actual_ticks"], 6)
                    if delay < 120 and result.initiation is not None:
                        self.assertTrue(result.initiation.accepted)
                        self.assertEqual(result.initiation.request.source_tick, 0)
                        self.assertEqual(result.decision.source_age_ticks, delay)
                        break
                    if delay >= 120 and result.decision.reason == "retired_reply":
                        break
                if delay < 120:
                    self.assertEqual(session.router.accepted, 1)
                else:
                    self.assertEqual(session.router.accepted, 0)
                    self.assertTrue(any(update.reason == "source_expired" for update in updates))
                    self.assertTrue(any(update.reason == "retired_reply" for update in updates))
                self.assertEqual(backend.max_inflight, 1)
                self.assertEqual(backend.calls, 1)
                session.revoke("case_complete")

    def test_human_takeover_drives_local_feedback_while_unacknowledged_model_remains_busy(self):
        # Hold the owner clock at the control boundary for the positive takeover
        # contract. Realtime scheduling cannot promise three fresh native leases.
        session, backend = self.session(240, model="pouncer",
                                        cancellation_ack=False, outage=True)
        session.step()
        before = session.view["tick"]
        receipt = session.submit_intent("human", "charge_hold", {}, request_id="human-charge")
        self.assertTrue(receipt.accepted)
        self.assertEqual(session.router.grant.source, "human")
        self.assertLessEqual(session.view["tick"] - before, 6)
        results = [session.step() for _ in range(3)]
        self.assertTrue(all(result.progress.action.primary for result in results))
        self.assertGreater(session.view["charge"], 0)
        self.assertEqual(backend.calls, 1)
        self.assertIsNotNone(session.router._backend)
        session.revoke("human_release_ownership")
        self.assertEqual(session.view["charge"], 0)
        result = session.step()
        self.assertTrue(all(tick["observation"]["charge"] == 0 for tick in result.info["trace"]))
        self.assertTrue(all(tick["observation"]["applied_impulse"] == [0, 0] for tick in result.info["trace"]))

    def test_jitter_and_repeated_model_grants_keep_original_request_age(self):
        # Provider jitter is measured at exact stepped control boundaries.
        session, backend = self.session(6, jitter_ticks=(-2, 3), ticks=2400)
        ages = []
        for effective in (4, 9):
            for _ in range(5):
                result = session.step()
                if result.initiation is not None:
                    self.assertTrue(result.initiation.accepted)
                    ages.append(result.decision.source_age_ticks)
                    self.assertGreaterEqual(ages[-1], effective)
                    self.assertLess(ages[-1], effective + 12)
                    break
            else:
                self.fail("jittered reply was not admitted in its bounded interval")
            session.revoke("explicit_new_invocation")
            session.transfer("model")
        self.assertEqual(backend.calls, 2)
        self.assertEqual(backend.max_inflight, 1)
        session.revoke()

    def test_reordered_and_duplicate_replies_are_refused_while_local_skill_continues(self):
        class ReorderedProvider(DelayedMockProvider):
            phase = 0
            duplicate = None
            def poll(self, tick):
                if self.phase == 0 and self._active is not None and tick >= self._active[1]:
                    self.phase = 1
                    return "older-retired-request", mock_proposal(self._active[0], "aim_use", settle_ticks=60)
                if self.phase == 2:
                    self.phase = 3
                    return self.duplicate
                value = super().poll(tick)
                if value is not None:
                    self.duplicate, self.phase = value, 2
                return value
        session, _ = self.session(6)
        backend = ReorderedProvider(lambda req: mock_proposal(req, "aim_use", settle_ticks=60), delay_ticks=6)
        session.router.provider = backend
        first = session.step()
        wrong = session.step()
        accepted = session.step()
        duplicate = session.step()
        self.assertEqual(first.decision.reason, "requested")
        self.assertEqual(wrong.decision.reason, "request_mismatch")
        self.assertTrue(accepted.initiation.accepted)
        self.assertEqual(duplicate.decision.reason, "duplicate_or_unsolicited")
        self.assertEqual(duplicate.progress.status, "running")
        self.assertEqual(session.router.accepted, 1)
        self.assertEqual(backend.calls, 1)
        self.assertGreater(duplicate.observation["tick"], accepted.observation["tick"])
        session.revoke()

    def test_expected_release_finishes_before_old_equipment_plan_is_retired(self):
        session, _ = self.session(30, model="hammerhead")
        session.submit_intent("script", "release", {}, request_id="own-release")
        session.step()
        result = session.step()
        self.assertEqual(result.progress.status, "succeeded")
        self.assertEqual(result.progress.reason, "local_detachment_confirmed")
        self.assertTrue(session.view["parts"][4]["detached"])
        self.assertEqual(session.identity().mount_generation, 1)
        cancels = len(session.cancellations)
        session.step()
        self.assertEqual(len(session.cancellations), cancels)
        session.transfer("model")
        self.assertEqual(session.router.grant.identity.mount_generation, 1)
        session.revoke()

    def test_active_model_plan_caps_motor_leases_and_expires_before_long_skill_timeout(self):
        session, backend = self.session(0, model="pouncer")
        backend.choose = lambda req: mock_proposal(req, "charge_hold", timeout_ticks=240)
        expires = None
        for _ in range(23):
            result = session.step()
            if result.initiation is not None:
                expires = result.initiation.request.source_tick + 120
            if result.decision.reason == "smart_plan_expired":
                self.assertEqual(session.view["charge"], 0)
                self.assertIsNone(session.router.grant)
                break
            for tick in result.info["trace"]:
                for packet in tick["requested"]:
                    self.assertLessEqual(packet["observed_tick"] + packet["lease_ticks"], expires)
        else:
            self.fail("an active model skill acquired a fresh plan lifetime")
        self.assertEqual(backend.calls, 1)

        # The explicit typed local mission retains its declared 240-tick bound.
        session.reset()
        session.submit_intent("script", "charge_hold", {"timeout_ticks": 240})
        for _ in range(22):
            result = session.step()
            self.assertEqual(result.progress.status, "running")
        self.assertGreater(session.view["tick"], 120)
        self.assertGreater(session.view["charge"], 0)
        session.revoke("typed_mission_complete")
        self.assertEqual(session.view["charge"], 0)

    def test_external_attachment_change_and_full_reset_invalidate_pending_reply(self):
        session, backend = self.session(30, model="hammerhead", cancellation_ack=False)
        session.step()
        old_run = session.view["run_id"]
        session.env.step({"release_usable": True})
        result = session.step()
        self.assertEqual(result.decision.reason, "lifetime_changed")
        self.assertEqual(session.router.accepted, 0)
        self.assertEqual(session.identity().mount_generation, 1)
        session.reset(seed=74)
        self.assertNotEqual(session.view["run_id"], old_run)
        self.assertEqual(session.identity().mount_generation, 0)
        for _ in range(7):
            session.step()
        self.assertEqual(session.router.accepted, 0)
        self.assertEqual(backend.calls, 1)

    def test_unconfirmed_cancel_never_sends_neutral_release_and_reset_recovers(self):
        session, _ = self.session(240, model="pouncer", cancellation_ack=False, outage=True)
        session.submit_intent("script", "charge_hold", {})
        session.step()
        original_cancel = session.env.cancel
        session.env.cancel = lambda: (_ for _ in ()).throw(RuntimeError("injected cancel outage"))
        with self.assertRaisesRegex(RuntimeError, "cancel outage"):
            session.revoke("takeover")
        with self.assertRaises(ContractError) as raised:
            session.step()
        self.assertEqual(raised.exception.code, "owner_cleanup_failed")
        self.assertIsNone(session.router.grant)
        session.env.cancel = original_cancel
        session.reset()
        self.assertEqual(session.view["charge"], 0)
        self.assertEqual(session.step().decision.reason, "no_owner")

    def test_final_native_effect_remains_audited_while_unadjudicated_skill_is_censored(self):
        session, backend = self.session(0, ticks=12)
        backend.choose = lambda req: mock_proposal(req, "aim_use", aim=[0, -1], settle_ticks=1)
        first = session.step()
        self.assertFalse(first.truncated)
        self.assertEqual(first.progress.status, "running")
        final = session.step()
        self.assertTrue(final.truncated)
        self.assertEqual(final.progress.status, "cancelled")
        self.assertEqual(final.progress.reason, "host_terminal_censored")
        self.assertIsNone(final.progress.local_success)
        self.assertIsNone(final.progress.evaluator_success)
        effects = [effect for tick in final.info["trace"]
                   for effect in tick["observation"]["last_effects"]]
        self.assertTrue(any(effect["consumed"] and effect["changed_cells"] > 0 for effect in effects))
        self.assertEqual(final.observation["tick"], 12)
        again = session.step()
        self.assertEqual(again.info["actual_ticks"], 0)
        self.assertEqual(again.progress, final.progress)
        with self.assertRaises(ContractError) as raised:
            session.transfer("model")
        self.assertEqual(raised.exception.code, "host_terminal")
        self.assertIsNone(session.router.grant)


@unittest.skipUnless(os.environ.get("NCA_HOST_CHECKOUT") and os.environ.get("NCA_GODOT"),
                     "actual owner is not configured; no asynchronous game qualification")
class ContinuousDecisionTests(_DecisionCases, unittest.TestCase):
    """Independent wall-clock advance, native expiry and delayed terminal readback."""

    mode = "realtime"

    def test_all_declared_delays_keep_advancing_and_expiry_is_source_anchored(self):
        for delay in (0, 6, 30, 120, 240):
            with self.subTest(delay=delay):
                # Keep the native terminal well beyond this lease assertion. The
                # realtime host advances while the controller thread can be descheduled
                # on shared CI; terminal arrival must not masquerade as lease behavior.
                session, backend = self.session(delay, cancellation_ack=False, ticks=2400)
                updates = []
                results = []
                started = session.view["tick"]
                for _ in range(44):
                    result = session.step()
                    results.append(result)
                    updates.append(result.decision)
                    self.assertEqual(result.info["actual_ticks"], 6)
                    self.assertTrue(result.info["consumed_trace_complete"])
                    if delay < 120 and result.initiation is not None:
                        self.assertTrue(result.initiation.accepted)
                        proposal = result.decision.proposal
                        self.assertEqual(result.initiation.request.source_tick, proposal.observed_tick)
                        self.assertEqual(result.initiation.request.source_observation_id, proposal.observation_id)
                        self.assertGreaterEqual(result.decision.source_age_ticks, delay)
                        self.assertLess(result.decision.source_age_ticks, 120)
                        if delay == 0:
                            self.assertEqual(result.decision.source_age_ticks, 0)
                        break
                    if delay >= 120 and result.decision.reason == "retired_reply":
                        break
                    if result.decision.reason == "lifetime_changed":
                        break
                self.assertGreater(session.view["tick"], started)
                expiries = self.native_expiries(results, session.cancellations)
                if session.router.grant is None:
                    self.assertTrue(expiries, "lost realtime ownership needs consecutive native expiry receipts")
                    self.assertEqual(session.router.accepted, 0)
                    self.assertIsNotNone(session.router._backend, "native expiry cannot free unacknowledged work")
                    self.assertTrue(session.router._retired)
                    self.assertTrue(any(update.reason == "lifetime_changed" for update in updates))
                elif delay >= 120:
                    self.assertEqual(session.router.accepted, 0)
                    self.assertTrue(any(update.reason == "source_expired" for update in updates))
                    self.assertTrue(any(update.reason == "retired_reply" for update in updates))
                else:
                    self.assertEqual(session.router.accepted, 1)
                self.assertEqual(backend.max_inflight, 1)
                self.assertEqual(backend.calls, 1)
                session.revoke("case_complete")

    def test_realtime_human_takeover_disarms_on_receipted_native_lease_expiry(self):
        session, backend = self.session(240, model="pouncer", ticks=2400,
                                        cancellation_ack=False, outage=True)
        results = [session.step()]
        receipt = session.submit_intent("human", "charge_hold", {}, request_id="realtime-human-charge")
        self.assertTrue(receipt.accepted)
        self.assertEqual(session.router.grant.source, "human")
        generation = receipt.request.actor.control_generation
        self.assertEqual(backend.calls, 1)
        self.assertIsNotNone(session.router._backend)
        charging = session.step()
        results.append(charging)
        self.assertEqual(charging.progress.status, "running")
        self.assertTrue(charging.progress.action.primary)
        self.assertTrue(any(tick["observation"]["charge"] > 0 for tick in charging.info["trace"]))
        # A real control-thread delay must not silently grant a fresh lifetime.
        # Keep it below the 36-tick receipt capacity and prove expiry from ticks,
        # source lease, and control generations, rather than elapsed wall time.
        time.sleep(.25)
        result = session.step()
        results.append(result)
        self.assertTrue(result.info["consumed_trace_complete"])
        self.assertEqual(result.progress.status, "invalidated")
        self.assertEqual(result.progress.reason, "actor_lifetime_changed")
        self.assertIn(generation + 1, self.native_expiries(results, session.cancellations))
        self.assertIsNone(session.router.grant)
        self.assertFalse(result.progress.action.primary)
        self.assertEqual(result.observation["charge"], 0)
        self.assertEqual(backend.calls, 1)
        self.assertIsNotNone(session.router._backend)
        session.revoke("human_release_ownership")
        self.assertEqual(session.view["charge"], 0)
        result = session.step()
        self.assertTrue(all(tick["observation"]["charge"] == 0 for tick in result.info["trace"]))
        self.assertTrue(all(tick["observation"]["applied_impulse"] == [0, 0] for tick in result.info["trace"]))

    def test_short_inference_pause_uses_idle_receipts_and_terminal_does_not_step_again(self):
        session, _ = self.session(240, ticks=18, cancellation_ack=False)
        session.step()
        time.sleep(.35)
        result = session.step()
        self.assertTrue(result.truncated)
        self.assertEqual(result.decision.reason, "host_terminal")
        self.assertEqual(result.observation["tick"], 18)
        self.assertGreater(len(result.info["idle_trace"]), 0)
        self.assertEqual(result.info["actual_ticks"], 0)
        self.assertEqual(session.router.accepted, 0)


if __name__ == "__main__":
    unittest.main()
