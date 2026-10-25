import unittest
from copy import deepcopy

from nca.adapters.flight import (
    ACTION_PROFILE,
    ACTION_VERSION,
    ACTOR_ID,
    CONTROLLER_SHA256,
    COURSE_VERSION,
    DECISION_TICKS,
    EPISODE_VERSION,
    FLAT_CHANNELS,
    FLAT_OBSERVATION_VERSION,
    HOST_SCHEDULE,
    LEASE_TICKS,
    OBSERVATION_VERSION,
    PHYSICS_HZ,
    SCENARIO_MAX_TICKS,
    SPLIT_GEOMETRY_VERSION,
    TRAINER_PROFILE,
    TRAINER_PROFILE_SHA256,
    TEACHER_SHA256,
    FlightProtocolError,
    FlightSession,
    FlightTeacherCaptureSession,
    UnknownConsumptionError,
    finite_vector,
    velocity_action,
)
from nca.train.flight import export_preprocessing, training_admission


class Owner:
    """Software fixture shaped exactly like the accepted #229 owner bridge."""

    def __init__(self):
        self.run = 0
        self.tick = 0
        self.epoch = 0
        self.generation = 0
        self.producer = "policy"
        self.closed = False
        self.requests = []
        self.mutate_reply = None
        self.partial_ticks = None
        self.step_failure = None
        self.aborts = []
        self.lease_deadline = 0

    def abort_after_reply(self, reason):
        self.aborts.append(reason)

    def observation(self):
        return {
            "version": OBSERVATION_VERSION,
            "action_version": ACTION_VERSION,
            "run_id": f"run-{self.run}",
            "actor_id": ACTOR_ID,
            "epoch": self.epoch,
            "observation_id":
                f"run-{self.run}:{self.epoch}:{ACTOR_ID}:{self.generation}:{self.tick}",
            "observed_tick": self.tick,
            "control_generation": self.generation,
            "command_live": self.lease_deadline > self.tick,
            "lease_remaining_ticks": max(0, self.lease_deadline - self.tick),
            "actuator_mode": (
                "assisted" if self.lease_deadline > self.tick
                else "passive_no_lease"
            ),
            "assistance_valid": self.lease_deadline > self.tick,
        }

    def flat(self, observation=None):
        observation = self.observation() if observation is None else observation
        return {
            "version": FLAT_OBSERVATION_VERSION,
            "field_count": FLAT_CHANNELS,
            "values": [0.0] * FLAT_CHANNELS,
            "source_observation_id": observation["observation_id"],
        }

    def manifest(self, command):
        return {
            "schema": EPISODE_VERSION,
            "observation": OBSERVATION_VERSION,
            "action": ACTION_VERSION,
            "flat_observation": FLAT_OBSERVATION_VERSION,
            "flat_observation_length": FLAT_CHANNELS,
            "action_profile": ACTION_PROFILE,
            "source": command["producer"],
            "course_version": COURSE_VERSION,
            "split_geometry_version": SPLIT_GEOMETRY_VERSION,
            "course": command["scenario"],
            "seed": command["seed"],
            "split": command["split"],
            "host_schedule": HOST_SCHEDULE,
            "physics_hz": PHYSICS_HZ,
            "decision_ticks": DECISION_TICKS,
            "lease_ticks": LEASE_TICKS,
            "profile_hash": TRAINER_PROFILE_SHA256,
            "controller_sha256": CONTROLLER_SHA256,
            "native_coupling": True,
            "host_count": 1,
            "sealed_suite_sha256": None,
            "max_ticks": SCENARIO_MAX_TICKS[command["scenario"]],
            "teacher_source_sha256":
                TEACHER_SHA256 if command["producer"] == "teacher" else None,
        }

    def transition(
        self, *, ticks, terminated=False, truncated=False, failed=False,
        success=False, reason="running"
    ):
        return {
            "version": EPISODE_VERSION,
            "ticks_consumed": ticks,
            "physical_ticks_total": self.tick,
            "observation": self.observation(),
            "terminated": terminated,
            "truncated": truncated,
            "failed": failed,
            "reason": reason,
            "success": success,
            "navigation_credit": success,
            "reward": 0.0,
            "reward_terms_last_tick": {},
            "episode_done": terminated or truncated or failed,
            "waypoint": 0,
            "dwell_ticks": 0,
        }

    def request(self, command):
        self.requests.append(deepcopy(command))
        expected_sequence = len(self.requests)
        if command["sequence"] != expected_sequence:
            raise AssertionError(
                "fixture received nonsequential client command"
            )

        operation = command["command"]
        if operation == "reset":
            if (
                command["profile"] != TRAINER_PROFILE
                or command["producer"] not in ("policy", "teacher")
            ):
                raise AssertionError("fixture scope drift")
            if (
                command["producer"] == "teacher"
                and command["split"] != "development"
            ):
                raise AssertionError("teacher capture split drift")
            if command["split"] not in ("development", "tuning"):
                raise AssertionError(
                    "fixture received excluded split"
                )
            self.run += 1
            self.epoch += 1
            self.tick = 0
            self.generation = 0
            self.lease_deadline = 0
            self.producer = command["producer"]
            observation = self.observation()
            reply = {
                "sequence": command["sequence"],
                "consumption": "confirmed",
                "manifest": self.manifest(command),
                "transition": self.transition(ticks=0),
                "flat_observation": self.flat(observation),
            }
        elif operation == "step":
            current = self.observation()
            for request_key, observation_key in (
                ("run_id", "run_id"),
                ("actor_id", "actor_id"),
                ("epoch", "epoch"),
                ("control_generation", "control_generation"),
                ("observation_id", "observation_id"),
                ("physical_tick", "observed_tick"),
            ):
                if command[request_key] != current[observation_key]:
                    raise AssertionError(
                        f"fixture authority mismatch: {request_key}"
                    )
            repeat = command["repeat_ticks"]
            if (
                type(repeat) is not int
                or not 1 <= repeat <= DECISION_TICKS
                or self.tick % DECISION_TICKS + repeat > DECISION_TICKS
            ):
                raise AssertionError("fixture cadence drift")
            action = command.get("action")
            if action is None:
                if self.lease_deadline <= self.tick:
                    raise AssertionError(
                        "fixture null continuation without live lease")
            else:
                if self.tick % DECISION_TICKS:
                    raise AssertionError(
                        "fixture fresh action outside decision boundary")
                self.lease_deadline = self.tick + LEASE_TICKS

            if self.step_failure == "expired_lease":
                reply = {
                    "sequence": command["sequence"],
                    "consumption": "confirmed",
                    "transition": self.transition(
                        ticks=0,
                        failed=True,
                        reason="action_admission: expired_observation",
                    ),
                    "flat_observation": self.flat(),
                }
            else:
                ticks = (
                    repeat
                    if self.partial_ticks is None
                    else min(repeat, self.partial_ticks)
                )
                previous_tick = self.tick
                self.tick += ticks
                if (
                    self.lease_deadline
                    and previous_tick < self.lease_deadline <= self.tick
                ):
                    self.lease_deadline = 0
                    self.generation += 1
                terminal = ticks < repeat
                reply = {
                    "sequence": command["sequence"],
                    "consumption": "confirmed",
                    "transition": self.transition(
                        ticks=ticks,
                        terminated=terminal,
                        success=terminal,
                        reason=(
                            "low_speed_arrival_dwell"
                            if terminal
                            else "running"
                        ),
                    ),
                    "flat_observation": self.flat(),
                }
        elif operation == "teacher_step":
            current = self.observation()
            if self.producer != "teacher":
                raise AssertionError(
                    "fixture teacher step requires teacher owner"
                )
            for request_key, observation_key in (
                ("run_id", "run_id"),
                ("actor_id", "actor_id"),
                ("epoch", "epoch"),
                ("control_generation", "control_generation"),
                ("observation_id", "observation_id"),
                ("physical_tick", "observed_tick"),
            ):
                if command[request_key] != current[observation_key]:
                    raise AssertionError(
                        f"fixture authority mismatch: {request_key}"
                    )
            if command["repeat_ticks"] != DECISION_TICKS:
                raise AssertionError("fixture cadence drift")
            requested = [0.25, -0.5]
            if self.step_failure == "teacher_refused":
                reply = {
                    "sequence": command["sequence"],
                    "consumption": "confirmed",
                    "transition": self.transition(
                        ticks=0,
                        failed=True,
                        reason="action_admission: expired_observation",
                    ),
                    "flat_observation": self.flat(),
                    "teacher_action": {
                        "source_sha256": TEACHER_SHA256,
                        "observation_id": current["observation_id"],
                        "status": "refused",
                        "requested_action": requested,
                        "admitted_action": None,
                        "reason": "action_admission: expired_observation",
                    },
                }
            else:
                ticks = (
                    DECISION_TICKS
                    if self.partial_ticks is None
                    else self.partial_ticks
                )
                self.tick += ticks
                terminal = ticks < DECISION_TICKS
                reply = {
                    "sequence": command["sequence"],
                    "consumption": "confirmed",
                    "transition": self.transition(
                        ticks=ticks,
                        terminated=terminal,
                        success=terminal,
                        reason=(
                            "low_speed_arrival_dwell"
                            if terminal
                            else "running"
                        ),
                    ),
                    "flat_observation": self.flat(),
                    "teacher_action": {
                        "source_sha256": TEACHER_SHA256,
                        "observation_id": current["observation_id"],
                        "status": "accepted",
                        "requested_action": requested,
                        "admitted_action": requested,
                        "reason": "owner_reference_action_admitted",
                    },
                }
        elif operation in ("cancel", "takeover"):
            current = self.observation()
            for request_key, observation_key in (
                ("run_id", "run_id"),
                ("actor_id", "actor_id"),
                ("epoch", "epoch"),
                ("control_generation", "control_generation"),
                ("observation_id", "observation_id"),
                ("physical_tick", "observed_tick"),
            ):
                if command[request_key] != current[observation_key]:
                    raise AssertionError(
                        f"fixture authority mismatch: {request_key}"
                    )
            self.generation += 1
            self.producer = command["producer"]
            authority = self.observation()
            authority["producer"] = self.producer
            reply = {
                "sequence": command["sequence"],
                "consumption": "confirmed",
                "authority": authority,
            }
        elif operation == "close":
            self.closed = True
            reply = {
                "sequence": command["sequence"],
                "consumption": "confirmed",
                "closed": True,
            }
        else:
            raise AssertionError(
                f"unexpected operation {operation}"
            )

        if self.mutate_reply is not None:
            self.mutate_reply(reply)
        return reply

    def close(self):
        self.closed = True


class FlightTests(unittest.TestCase):
    def reset(self, owner=None):
        owner = Owner() if owner is None else owner
        session = FlightSession(owner)
        observation = session.reset(
            scenario="arrival_braking", seed=101
        )
        return owner, session, observation

    def test_reset_uses_real_trainer_identity_and_open_host_split(self):
        owner, session, observation = self.reset()
        command = owner.requests[0]
        self.assertEqual(command["profile"], "4ly_trainer")
        self.assertEqual(command["producer"], "policy")
        self.assertEqual(command["split"], "development")
        self.assertEqual(observation["actor_id"], 0)
        self.assertEqual(observation["observed_tick"], 0)
        self.assertEqual(len(session.flat_observation), 54)
        self.assertEqual(
            session.last_exchange["physical_tick_after"], 0
        )

    def test_preregistered_cases_and_source_deadlines_are_enforced_before_transport(self):
        invalid = (
            ("training", "neutral_control", 101),
            ("training", "arrival_braking", 1),
            ("tuning", "unsupported_hover", 211),
            ("tuning", "arrival_braking", 101),
        )
        for split, scenario, seed in invalid:
            owner = Owner()
            session = FlightSession(owner)
            with self.assertRaises(ValueError):
                session.reset(
                    scenario=scenario, seed=seed, split=split
                )
            self.assertEqual(owner.requests, [])

        owner = Owner()
        session = FlightSession(owner)
        session.reset(
            scenario="slalom", seed=109, split="training"
        )
        self.assertEqual(
            session.manifest["max_ticks"], 2400
        )
        self.assertEqual(
            owner.requests[0]["split"], "development"
        )

    def test_partial_step_null_continuation_and_real_lease_expiry(self):
        owner = Owner()
        session = FlightSession(owner)
        session.reset(
            scenario="horizontal_reach", seed=101, split="training")
        session.step(
            [0.25, 0.0],
            observation_id=session.observation["observation_id"],
            repeat_ticks=6,
        )
        session.step(
            [0.25, 0.0],
            observation_id=session.observation["observation_id"],
            repeat_ticks=3,
        )
        session.continue_step(
            observation_id=session.observation["observation_id"],
            repeat_ticks=3,
        )
        session.continue_step(
            observation_id=session.observation["observation_id"],
            repeat_ticks=5,
        )
        before_generation = session.observation["control_generation"]
        reply = session.continue_step(
            observation_id=session.observation["observation_id"],
            repeat_ticks=1,
        )
        self.assertEqual(reply["transition"]["physical_ticks_total"], 18)
        self.assertEqual(
            session.observation["control_generation"],
            before_generation + 1,
        )
        self.assertFalse(session.observation["command_live"])
        self.assertEqual(session.observation["lease_remaining_ticks"], 0)
        self.assertEqual(
            session.observation["actuator_mode"], "passive_no_lease")
        self.assertFalse(session.observation["assistance_valid"])
        steps = [
            request for request in owner.requests
            if request["command"] == "step"
        ]
        self.assertEqual(
            [item["repeat_ticks"] for item in steps], [6, 3, 3, 5, 1])
        self.assertEqual(
            [item["action"] is None for item in steps],
            [False, False, True, True, True],
        )

    def test_teacher_capture_is_isolated_owner_reference_path(self):
        owner = Owner()
        session = FlightTeacherCaptureSession(owner)
        observation = session.reset(
            scenario="slalom", seed=109
        )
        self.assertEqual(
            owner.requests[0]["producer"], "teacher"
        )
        self.assertEqual(
            owner.requests[0]["split"], "development"
        )
        self.assertEqual(
            session.teacher_source_sha256, TEACHER_SHA256
        )
        with self.assertRaisesRegex(
            FlightProtocolError,
            "owner reference_action",
        ):
            session.step(
                [0.0, 0.0],
                observation_id=observation["observation_id"],
            )
        self.assertEqual(len(owner.requests), 1)
        reply = session.step_reference()
        self.assertEqual(
            reply["teacher_action"]["observation_id"],
            observation["observation_id"],
        )
        self.assertEqual(
            reply["teacher_action"]["requested_action"],
            [0.25, -0.5],
        )
        self.assertEqual(
            reply["teacher_action"]["admitted_action"],
            [0.25, -0.5],
        )
        self.assertEqual(
            reply["transition"]["ticks_consumed"], 6
        )
        self.assertFalse(session.terminal)
        self.assertFalse(
            hasattr(FlightSession(Owner()), "step_reference")
        )

    def test_teacher_refusal_is_retained_as_zero_tick_failed_interval(self):
        owner = Owner()
        session = FlightTeacherCaptureSession(owner)
        observation = session.reset(
            scenario="arrival_braking", seed=101
        )
        owner.step_failure = "teacher_refused"
        reply = session.step_reference()
        self.assertEqual(
            reply["teacher_action"]["status"], "refused"
        )
        self.assertIsNone(
            reply["teacher_action"]["admitted_action"]
        )
        self.assertTrue(reply["transition"]["failed"])
        self.assertEqual(
            reply["transition"]["ticks_consumed"], 0
        )
        self.assertEqual(
            session.observation["observation_id"],
            observation["observation_id"],
        )
        self.assertTrue(session.terminal)

    def test_teacher_capture_rejects_tuning_final_and_foreign_teacher_receipt(self):
        for split in ("tuning", "final", "sealed"):
            owner = Owner()
            session = FlightTeacherCaptureSession(owner)
            with self.assertRaises(ValueError):
                session.reset(
                    scenario="arrival_braking",
                    seed=211 if split == "tuning" else 101,
                    split=split,
                )
            self.assertEqual(owner.requests, [])

        owner = Owner()
        session = FlightTeacherCaptureSession(owner)
        session.reset(
            scenario="arrival_braking", seed=101
        )
        owner.mutate_reply = lambda reply: (
            reply["teacher_action"].update(
                observation_id="foreign"
            )
            if "teacher_action" in reply else None
        )
        with self.assertRaises(FlightProtocolError):
            session.step_reference()
        self.assertTrue(session.failed)

    def test_final_reset_and_invalid_actions_never_reach_owner(self):
        owner = Owner()
        session = FlightSession(owner)
        for split in (
            "final", "sealed", "evaluation",
            "original_opened_final", "polar_sealed_final",
        ):
            with self.assertRaises(ValueError):
                session.reset(
                    scenario="x", seed=1, split=split
                )
        self.assertEqual(owner.requests, [])
        for action in (
            [True, 0],
            [float("nan"), 0],
            [2, 0],
            [0],
        ):
            with self.assertRaises(ValueError):
                velocity_action(action)

    def test_actual_terminal_stops_at_first_partial_tick_and_preserves_final_observation(self):
        owner, session, observation = self.reset()
        owner.partial_ticks = 3
        reply = session.step(
            [1, 1],
            observation_id=observation["observation_id"],
        )
        self.assertEqual(
            reply["transition"]["ticks_consumed"], 3
        )
        self.assertEqual(
            reply["transition"]["observation"]["observed_tick"], 3
        )
        self.assertTrue(session.terminal)
        self.assertEqual(
            session.observation["observed_tick"], 3
        )
        self.assertEqual(
            session.last_exchange["ticks_consumed"], 3
        )
        with self.assertRaises(FlightProtocolError):
            session.step(
                [0, 0],
                observation_id=session.observation[
                    "observation_id"
                ],
            )
        self.assertEqual(len(owner.requests), 2)

    def test_full_step_checks_physical_tick_and_observation_identity(self):
        owner, session, observation = self.reset()
        reply = session.step(
            [0.5, -0.25],
            observation_id=observation["observation_id"],
        )
        self.assertEqual(
            reply["transition"]["physical_ticks_total"], 6
        )
        self.assertEqual(
            session.observation["observed_tick"], 6
        )
        self.assertNotEqual(
            session.observation["observation_id"],
            observation["observation_id"],
        )
        self.assertFalse(session.terminal)

    def test_expired_lease_is_authoritative_zero_tick_failure_not_a_sample(self):
        owner, session, observation = self.reset()
        owner.step_failure = "expired_lease"
        reply = session.step(
            [0, 0],
            observation_id=observation["observation_id"],
        )
        self.assertTrue(reply["transition"]["failed"])
        self.assertEqual(
            reply["transition"]["ticks_consumed"], 0
        )
        self.assertEqual(
            session.observation["observed_tick"], 0
        )
        self.assertTrue(session.terminal)
        self.assertEqual(
            session.last_exchange["ticks_consumed"], 0
        )

    def test_cancel_revokes_authority_without_reset_or_physical_tick(self):
        owner, session, observation = self.reset()
        authority = session.cancel()
        self.assertEqual(
            authority["run_id"], observation["run_id"]
        )
        self.assertEqual(
            authority["epoch"], observation["epoch"]
        )
        self.assertEqual(
            authority["observed_tick"],
            observation["observed_tick"],
        )
        self.assertEqual(
            authority["control_generation"],
            observation["control_generation"] + 1,
        )
        self.assertNotEqual(
            authority["observation_id"],
            observation["observation_id"],
        )
        self.assertEqual(authority["producer"], "policy")
        self.assertTrue(session.revoked)
        with self.assertRaises(FlightProtocolError):
            session.step(
                [0, 0],
                observation_id=observation["observation_id"],
            )

    def test_human_takeover_revokes_policy_without_reset(self):
        owner, session, observation = self.reset()
        authority = session.takeover()
        self.assertEqual(authority["producer"], "human")
        self.assertEqual(
            authority["observed_tick"],
            observation["observed_tick"],
        )
        self.assertEqual(
            authority["run_id"], observation["run_id"]
        )
        self.assertTrue(session.revoked)
        other = FlightSession(Owner())
        with self.assertRaises(ValueError):
            other.takeover("teacher")

    def test_reset_replaces_run_and_fences_previous_authority(self):
        owner, session, before = self.reset()
        after = session.reset(
            scenario="horizontal_reach",
            seed=211,
            split="tuning",
        )
        self.assertNotEqual(
            after["run_id"], before["run_id"]
        )
        self.assertNotEqual(
            after["epoch"], before["epoch"]
        )
        self.assertEqual(
            owner.requests[-1]["split"], "tuning"
        )
        with self.assertRaises(FlightProtocolError):
            session.step(
                [0, 0],
                observation_id=before["observation_id"],
            )

    def test_malformed_owner_and_flat_replies_poison_session(self):
        changes = [
            lambda reply:
                reply["transition"].update(version="legacy"),
            lambda reply:
                reply["transition"]["observation"].update(actor_id=1),
            lambda reply:
                reply["flat_observation"].update(field_count=53),
            lambda reply:
                reply["flat_observation"].update(
                    values=[float("inf")] * 54),
            lambda reply:
                reply["flat_observation"].update(
                    source_observation_id="foreign"),
            lambda reply:
                reply["manifest"].update(decision_ticks=5),
            lambda reply:
                reply["manifest"].update(max_ticks=960),
        ]
        for change in changes:
            owner = Owner()
            owner.mutate_reply = change
            session = FlightSession(owner)
            with self.assertRaises(
                (FlightProtocolError, ValueError)
            ):
                session.reset(scenario="arrival_braking", seed=101)
            self.assertTrue(session.failed)
            self.assertTrue(owner.aborts)
            self.assertIn(
                "reset_semantic_validation",
                owner.aborts[-1],
            )

    def test_duplicate_or_reordered_reply_is_unknown_consumption_and_never_replayed(self):
        for returned in (0, 2):
            owner = Owner()
            owner.mutate_reply = (
                lambda reply, value=returned:
                    reply.update(sequence=value)
            )
            session = FlightSession(owner)
            with self.assertRaises(
                UnknownConsumptionError
            ):
                session.reset(scenario="arrival_braking", seed=101)
            self.assertTrue(session.failed)
            self.assertEqual(owner.aborts, [])
            self.assertEqual(
                session.unknown_consumption["state"],
                "unknown",
            )
            with self.assertRaises(FlightProtocolError):
                session.reset(scenario="arrival_braking", seed=101)
            self.assertEqual(len(owner.requests), 1)

    def test_lost_step_reply_retains_unknown_consumption_and_refuses_replay(self):
        class LostAfterConsume(Owner):
            def request(self, command):
                if command["command"] != "step":
                    return super().request(command)
                self.requests.append(deepcopy(command))
                self.tick += DECISION_TICKS
                raise UnknownConsumptionError(
                    "step",
                    command["sequence"],
                    "fixture_dropped_reply",
                )

        owner, session, observation = self.reset(
            LostAfterConsume()
        )
        with self.assertRaises(UnknownConsumptionError):
            session.step(
                [0.5, 0],
                observation_id=observation["observation_id"],
            )
        self.assertTrue(session.failed)
        self.assertEqual(
            session.unknown_consumption["operation"],
            "step",
        )
        self.assertEqual(
            session.unknown_consumption["run_id"],
            observation["run_id"],
        )
        self.assertEqual(
            session.unknown_consumption["observation_id"],
            observation["observation_id"],
        )
        self.assertEqual(
            session.unknown_consumption["physical_tick"], 0
        )
        self.assertEqual(owner.tick, 6)
        with self.assertRaises(FlightProtocolError):
            session.step(
                [0.5, 0],
                observation_id=observation["observation_id"],
            )
        self.assertEqual(len(owner.requests), 2)

    def test_short_nonterminal_interval_refused(self):
        owner, session, observation = self.reset()
        owner.partial_ticks = 1

        def make_nonterminal(reply):
            reply["transition"].update(
                terminated=False,
                truncated=False,
                failed=False,
                success=False,
                episode_done=False,
            )

        owner.mutate_reply = make_nonterminal
        with self.assertRaises(FlightProtocolError):
            session.step(
                [0, 0],
                observation_id=observation["observation_id"],
            )
        self.assertTrue(session.failed)

    def test_close_is_explicit_and_idempotent(self):
        owner, session, _ = self.reset()
        session.close()
        self.assertTrue(owner.closed)
        close_count = sum(
            request["command"] == "close"
            for request in owner.requests
        )
        session.close()
        self.assertEqual(
            sum(
                request["command"] == "close"
                for request in owner.requests
            ),
            close_count,
        )

    def test_export_preprocessing_and_admission_remain_fail_closed(self):
        self.assertFalse(training_admission()["admitted"])
        flat, action = export_preprocessing(
            [0] * 54, [1, 1]
        )
        self.assertAlmostEqual(
            sum(x * x for x in action["move"]),
            1.0,
        )
        self.assertEqual(len(flat), 54)
        self.assertEqual(finite_vector(flat, 54), flat)


if __name__ == "__main__":
    unittest.main()
