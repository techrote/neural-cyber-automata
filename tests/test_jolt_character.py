"""Character metrics use physical receipts rather than controller labels."""

from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

from nca.contracts import ContractError, WireAction
from nca.data import read_episode
from nca.evaluation.jolt_character import character_metrics
from test_jolt_evidence_durability import bound_fixture


class CharacterMetricTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        _, _, rows = bound_fixture(Path(folder.name))
        self.episode = read_episode(Path(folder.name) / rows[0]["artifact"])
        self.initial = self.episode.initial_observation
        self.frames = [dict(frame) for frame in self.episode.evaluation["controller_frames"]]
        self.events = list(self.episode.events)

    def metrics(self, *, events=None, frames=None, initial=None):
        return character_metrics(self.initial if initial is None else initial, (50., 0.),
            self.events if events is None else events, self.frames if frames is None else frames)

    def with_action(self, events, frames, index, action):
        event = events[index]
        packet = {**event.requested_packet, "action": action}
        ticks = []
        for i, tick in enumerate(event.ticks):
            ticks.append(replace(tick, requested=(packet,) if i == 0 else (),
                consumed_request=packet, consumed_action=action,
                applied={**tick.applied, "action": action}))
        events[index] = replace(event, requested_packet=packet, ticks=tuple(ticks))
        frames[index] = {**frames[index], "action": action}

    def test_actual_impulses_remain_distinct_from_requested_movement(self):
        result = self.metrics()
        self.assertAlmostEqual(result["aggregate_linear_impulse_sum"][0], 1.2)
        self.assertEqual(result["aggregate_angular_impulse_sum"], 0.)
        self.assertEqual(result["requested_primary_edges"], 0)
        self.assertEqual(result["confirmed_primary_associated_pulses"], 0)
        self.assertEqual(result["airborne_ticks"], 0)
        self.assertIn("unmeasured", result["passive_contact_wrench"])

    def test_initial_fall_is_airborne_but_not_a_purposeful_launch(self):
        initial = {**self.initial, "grounded": False}
        events = [replace(event, ticks=tuple(replace(tick, observation={**tick.observation, "grounded": False})
                                            for tick in event.ticks)) for event in self.events]
        result = self.metrics(initial=initial, events=events)
        self.assertEqual(result["airborne_ticks"], 12)
        self.assertEqual(result["confirmed_primary_associated_pulses"], 0)
        self.assertEqual(result["pulses_with_airborne_confirmation"], 0)
        self.assertEqual(result["support_loss_transitions"], 0)

    def test_requested_and_admitted_hop_needs_consumed_edge_and_physical_pulse(self):
        events, frames = list(self.events), list(self.frames)
        action = WireAction(primary=True).to_dict()
        self.with_action(events, frames, 0, action)
        result = self.metrics(events=events, frames=frames)
        self.assertEqual(result["requested_primary_edges"], 1)
        self.assertEqual(result["admitted_primary_edges"], 1)
        self.assertEqual(result["consumed_primary_edges"], 1)
        self.assertEqual(result["confirmed_primary_associated_pulses"], 0)
        ticks = list(events[0].ticks)
        ticks[0] = replace(ticks[0], applied={**ticks[0].applied, "linear_impulse": [0., -2.], "energy_used": .1},
                           observation={**ticks[0].observation, "cooldown": .7, "grounded": False})
        events[0] = replace(events[0], ticks=tuple(ticks))
        result = self.metrics(events=events, frames=frames)
        self.assertEqual(result["confirmed_primary_associated_pulses"], 1)
        self.assertEqual(result["pulses_with_airborne_confirmation"], 1)
        self.assertEqual(result["pulse_ticks"], [1])

    def test_unavailable_impulse_and_support_are_not_replaced_with_zero(self):
        ticks = list(self.events[0].ticks)
        ticks[0] = replace(ticks[0], applied={"action_available": False},
                           observation={**ticks[0].observation, "grounded": None})
        events = [replace(self.events[0], ticks=tuple(ticks)), self.events[1]]
        result = self.metrics(events=events)
        self.assertIsNone(result["aggregate_linear_impulse_sum"])
        self.assertIsNone(result["aggregate_angular_impulse_sum"])
        self.assertEqual(result["unknown_linear_impulse_ticks"], 1)
        self.assertEqual(result["unknown_support_ticks"], 1)

    def test_neutral_host_assist_is_not_credited_to_a_later_policy_request(self):
        events, frames = list(self.events), list(self.frames)
        self.with_action(events, frames, 0, WireAction(effort=0.).to_dict())
        result = self.metrics(events=events, frames=frames)
        self.assertEqual(result["first_aggregate_actuation_seconds"], 1 / 60.)
        self.assertEqual(result["first_active_request_seconds"], .1)
        self.assertEqual(result["request_to_aggregate_actuation_ticks"], 1)
        self.assertIsNone(result["first_toward_motion_seconds"])

    def test_auxiliary_only_emergency_is_an_active_request(self):
        events, frames = list(self.events), list(self.frames)
        self.with_action(events, frames, 0, WireAction(effort=0.).to_dict())
        self.with_action(events, frames, 1, WireAction(auxiliary=True).to_dict())
        result = self.metrics(events=events, frames=frames)
        self.assertEqual(result["first_active_request_seconds"], .1)
        self.assertEqual(result["request_to_aggregate_actuation_ticks"], 1)

    def test_recovery_phase_needs_subsequent_physical_progress_to_count_response(self):
        frames = [dict(frame) for frame in self.frames]
        frames[0].update(phase="contact_recovery_anticipate", reason="supported_progress_stall",
            diagnostic_state={"recovery_epoch": 1, "supervisor_intervention": False})
        frames[1].update(phase="contact_recovery_launch", reason="fresh_contact",
            diagnostic_state={"recovery_epoch": 1, "supervisor_intervention": False})
        result = self.metrics(frames=frames)
        self.assertEqual(result["ordinary_recovery_events"], 1)
        self.assertEqual(result["supervisor_intervention_decisions"], 0)
        self.assertEqual(result["recoveries_with_observed_progress"], 1)
        self.assertEqual(result["recovery_events"][0]["actual_primary_pulses"], 0)
        still = [replace(event, ticks=tuple(replace(tick, observation={**tick.observation, "position": [0., 0.]})
                                            for tick in event.ticks)) for event in self.events]
        self.assertEqual(self.metrics(events=still, frames=frames)["recoveries_with_observed_progress"], 0)

    def test_emergency_supervisor_and_ordinary_contact_recovery_are_distinct(self):
        frames = [dict(frame) for frame in self.frames]
        frames[0].update(phase="emergency_recovery", diagnostic_state={"recovery_epoch": 1,
            "supervisor_intervention": True, "supervisor_reason": "declared_tilt_emergency"})
        frames[1].update(phase="emergency_recovery", diagnostic_state={"recovery_epoch": 1,
            "supervisor_intervention": True, "supervisor_reason": "declared_tilt_emergency"})
        result = self.metrics(frames=frames)
        self.assertEqual(result["ordinary_recovery_events"], 0)
        self.assertEqual(result["supervisor_recovery_events"], 1)
        self.assertEqual(result["supervisor_intervention_decisions"], 2)
        self.assertEqual(result["supervisor_interventions_by_reason"], {"declared_tilt_emergency": 2})

    def test_changed_controller_frame_cannot_replace_the_native_request(self):
        frames = [dict(frame) for frame in self.frames]
        frames[0]["action"] = WireAction(primary=True).to_dict()
        with self.assertRaisesRegex(ContractError, "character_frames"):
            self.metrics(frames=frames)
        with self.assertRaisesRegex(ContractError, "character_frames"):
            self.metrics(frames=frames[:-1])


if __name__ == "__main__":
    unittest.main()
