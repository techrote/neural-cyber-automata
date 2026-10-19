"""Boundary conformance, with explicit protocol-fixture evidence limits."""

import json
import math
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from nca.contracts import (
    ActuationReceipt, Admission, Assist, Channel, Command, ContractError,
    ControlCapabilities, EvaluatorRecord, GameplayProfile, Joint, LegacyCommand,
    LegacyMetalAction, LegacyWireAction, LifecycleStamp, Observation, Part, Sensor,
    WireAction, admit_command, canonical_json, consume_admission, sha256_json,
    strict_json_loads,
)
from nca.registry import Registry


HOST = "f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497"


def profile():
    return GameplayProfile(
        id="fixture-rig-v1", version=1, host_revision=HOST, representation="recorded_fixture",
        parts=(Part("body", "box"), Part("foot", "capsule")),
        joints=(Joint("leg", "body", "foot", "slide", -1, 1, "game_cells",
                      play=.05, compliance=.1, damping=.3),),
        sensors=(Sensor("velocity", "cells_per_second", "actor_local", -100, 100),
                 Sensor("global_target", "game_cells", "world", -1000, 1000, "global")),
        channels=(Channel("drive", "body"), Channel("reserved", "body", lower=0, upper=0)),
        channel_mask=(True, False), assist=Assist("bounded_gyro", 2, 0.4, 0.05, 0.1, 0.05))


def observation(p=None, **kwargs):
    p = p or profile()
    value = Observation("obs-0", 0, LifecycleStamp("fixture", "actor-0", 1), p.sha256,
                        (2.0, None), (True, False))
    return replace(value, **kwargs)


def command(p=None, obs=None, **kwargs):
    p = p or profile()
    obs = obs or observation(p)
    value = Command("command-0", obs.lifecycle, p.sha256, obs.id, obs.tick, 0, 12,
                    (.5, 0), (True, False))
    return replace(value, **kwargs)


class ContractAssertions:
    def assertCode(self, code, callable, *args, **kwargs):
        with self.assertRaises(ContractError) as caught:
            callable(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)


class ContractTests(ContractAssertions, unittest.TestCase):
    def test_profile_roundtrip_and_mutation_resistance(self):
        p = profile()
        self.assertEqual(GameplayProfile.from_json(p.to_json()), p)
        self.assertEqual(GameplayProfile.from_dict(p.to_dict()).sha256, p.sha256)
        changed = p.to_dict()
        changed["parts"][0]["id"] = "changed"
        self.assertEqual(p.parts[0].id, "body")
        with self.assertRaises(FrozenInstanceError):
            p.version = 2

    def test_json_rejects_duplicates_nonfinite_and_oversize(self):
        self.assertCode("duplicate_field", strict_json_loads, '{"version":1,"version":2}')
        self.assertCode("nonfinite", strict_json_loads, '{"x":NaN}')
        self.assertCode("invalid_json", strict_json_loads, '{"x":1e999}')
        self.assertCode("size_limit", strict_json_loads, '{"x":"too long"}', max_bytes=4)
        self.assertCode("invalid_type", canonical_json, {1: "ambiguous key"})

    def test_unknown_versions_fields_and_boolean_numbers(self):
        for field, value in (("schema_version", 2), ("version", True), ("base_hz", False)):
            with self.subTest(field=field):
                data = profile().to_dict()
                data[field] = value
                with self.assertRaises(ContractError):
                    GameplayProfile.from_dict(data)
        data = profile().to_dict()
        data["rotor_momentum_budget"] = 1
        self.assertCode("unknown_field", GameplayProfile.from_dict, data)

    def test_masks_topology_and_duplicate_ids(self):
        p = profile()
        self.assertCode("shape_mismatch", replace, p, channel_mask=(True,))
        self.assertCode("duplicate_id", replace, p, parts=(p.parts[0], p.parts[0]))
        self.assertCode("invalid_topology", replace, p,
                        joints=(replace(p.joints[0], child="absent"),))
        self.assertCode("part_cap", replace, p, part_cap=1)

    def test_d011_profile_hash_includes_play_assistance_and_response(self):
        p = profile()
        for changed in (replace(p, assist=replace(p.assist, effort_limit=3)),
                        replace(p, joints=(replace(p.joints[0], play=.1),)),
                        replace(p, channels=(replace(p.channels[0], response_seconds=.1), p.channels[1]))):
            self.assertNotEqual(changed.sha256, p.sha256)

    def test_missing_sensors_are_null_and_evaluator_data_is_separate(self):
        p = profile()
        observation(p).validate_profile(p)
        self.assertCode("invalid_mask", observation, p, values=(0, 0))
        leaked = observation(p, values=(2, 20), valid=(True, True))
        self.assertCode("privilege_violation", leaked.validate_profile, p)
        self.assertCode("view_mismatch", observation(p, view="global").validate_profile, p)
        eval_record = EvaluatorRecord("obs-0", 0, {"target": {"hidden_position": [4, 9]}})
        self.assertCode("unknown_field", Observation.from_dict, eval_record.to_dict())
        with self.assertRaises(TypeError):
            eval_record.metrics["target"]["hidden_position"][0] = 2

    def test_capability_axes_remain_independent(self):
        c = ControlCapabilities(decision_placement="central")
        self.assertEqual(c.perception, "local")
        self.assertEqual(c.coordination, "independent")
        self.assertEqual(c.training_privilege, "none")

    def test_wire_bounded_finite_and_strict_boolean(self):
        self.assertEqual(WireAction.from_json(WireAction().to_json()), WireAction())
        for kwargs in ({"move": (1, 1)}, {"effort": math.inf}, {"trim": math.nan},
                       {"primary": 1}, {"effort": True}, {"aim": (0, 0, 0)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ContractError):
                WireAction(**kwargs)

    def test_expiry_is_observation_anchored_and_exclusive(self):
        p, obs = profile(), observation()
        cmd = command(p, obs)
        cmd.validate_against(p, obs, 11, obs.lifecycle)
        self.assertCode("expired", cmd.validate_against, p, obs, 12, obs.lifecycle)
        self.assertCode("expired", cmd.validate_against, p, obs, 99, obs.lifecycle)
        self.assertCode("invalid_clock", cmd.validate_against, p, obs, -1, obs.lifecycle)
        late = admit_command(cmd, p, obs, 12, obs.lifecycle)
        self.assertFalse(late.accepted)
        self.assertIsNone(late.admitted_action)
        self.assertEqual(late.reason, "expired")

    def test_slot_reuse_takeover_morphology_and_mount_changes_are_fenced(self):
        p, obs = profile(), observation()
        cmd = command(p, obs)
        for field in ("epoch", "incarnation", "control_generation", "morphology_version", "mount_generation"):
            with self.subTest(field=field):
                stamp = replace(obs.lifecycle, **{field: getattr(obs.lifecycle, field) + 1})
                self.assertCode("lifecycle_mismatch", cmd.validate_against, p, obs, 1, stamp)

    def test_lane_sequence_observation_and_reserved_channels_fail_closed(self):
        p, obs = profile(), observation()
        self.assertCode("sequence_replay", command().validate_against, p, obs, 1,
                        obs.lifecycle, last_sequence=0)
        for changed, code in ((command(lane="direct"), "lane_mismatch"),
                              (command(observation_id="invented"), "observation_mismatch"),
                              (command(action_mask=(True, True)), "shape_mismatch"),
                              (command(profile_sha256="0" * 64), "profile_mismatch")):
            self.assertCode(code, changed.validate_against, p, obs, 1, obs.lifecycle)

    def test_consumption_rechecks_lifetime_and_separates_motion(self):
        p, obs = profile(), observation()
        admitted = admit_command(command(), p, obs, 0, obs.lifecycle)
        receipt = consume_admission(admitted, p, obs, 1, obs.lifecycle, (.2, 0),
                                    passive_motion=(0, 3), limiting_flags=("contact_limited",))
        self.assertEqual(receipt.admission.requested.action, (.5, 0))
        self.assertEqual(receipt.consumed_action, (.5, 0))
        self.assertEqual(receipt.applied_effort, (.2, 0))
        self.assertEqual(receipt.passive_motion, (0, 3))
        self.assertCode("lifecycle_mismatch", consume_admission, admitted, p, obs, 1,
                        replace(obs.lifecycle, control_generation=1), (.2, 0))
        self.assertCode("expired", consume_admission, admitted, p, obs, 12, obs.lifecycle, (.2, 0))
        self.assertCode("invalid_receipt", replace, receipt, consumed_action=(.1, 0))
        self.assertCode("out_of_range", consume_admission,
                        replace(admitted, admitted_action=(2, 0)), p, obs, 1, obs.lifecycle, (.2, 0))
        self.assertCode("invalid_receipt", ActuationReceipt,
                        Admission(command(), False, "refused", 0), 1, (.5, 0), (.5, 0))

    def test_profile_fixture_files_are_current(self):
        root = Path(__file__).resolve().parents[1] / "configs" / "profiles"
        files = sorted(root.glob("*.json"))
        self.assertGreaterEqual(len(files), 4)
        for path in files:
            with self.subTest(profile=path.name):
                p = GameplayProfile.from_json(path.read_bytes())
                self.assertIn(p.representation, ("sampled_ACT003", "recorded_fixture"))


class LegacyContractTests(ContractAssertions, unittest.TestCase):
    def packet(self, action=None, kind="rig_action"):
        return LegacyCommand(1, "legacy", 0, 1, 0, 0, "issued-0", 12, kind,
                             action or LegacyWireAction.neutral(), False, (), "none", False)

    def test_v1_roundtrip_preserves_shape_and_diagonal_components(self):
        packet = self.packet(replace(LegacyWireAction.neutral(), movement=(1, 1)))
        self.assertEqual(LegacyCommand.from_dict(packet.to_dict()), packet)
        self.assertIn("movement", packet.to_dict()["action"])
        self.assertNotIn("move", packet.to_dict()["action"])
        data = packet.to_dict()
        data["version"] = 1.0
        self.assertEqual(LegacyCommand.from_dict(data), packet)
        data["version"] = 2
        with self.assertRaises(ContractError):
            LegacyCommand.from_dict(data)

    def test_v1_rejects_v2_flags_and_incomplete_fields(self):
        data = self.packet().to_dict()
        data["action"]["tool"] = False
        with self.assertRaises(ContractError):
            LegacyCommand.from_dict(data)
        data = self.packet().to_dict()
        del data["cancel"]
        self.assertCode("missing_field", LegacyCommand.from_dict, data)

    def test_v1_direct_mapping_clamps_without_overwriting_requested_values(self):
        action = LegacyMetalAction((-.5, -.3, .0, .0), (0, 0))
        self.assertEqual(action.admitted_actuators("gemini"), (0, 0, 0, 0))
        self.assertEqual(action.actuators, (-.5, -.3, 0, 0))
        self.assertEqual(LegacyCommand.from_json(self.packet(action, "actuator_action").to_json()).action,
                         action)
        self.assertCode("unsupported_channel", LegacyMetalAction((0, 0, 1, 0), (0, 0)).admitted_actuators,
                        "gemini")
        self.assertCode("unsupported_channel", LegacyMetalAction((0, 0, 0, .1), (0, 0)).admitted_actuators,
                        "ripper")
        self.assertCode("unsupported_rig", action.admitted_actuators, "jolt")

    def test_v1_cancel_cannot_release_or_fire(self):
        self.assertTrue(replace(self.packet(), cancel=True).cancel)
        self.assertCode("invalid_cancel", replace, self.packet(), cancel=True, tag=True)
        self.assertCode("invalid_cancel", replace, self.packet(), cancel=True,
                        action=replace(LegacyWireAction.neutral(), primary=True))
        self.assertCode("message_mismatch", replace, self.packet(), message_type="goal_hint")


class RegistryTests(ContractAssertions, unittest.TestCase):
    def test_duplicate_version_and_capability_refusals_precede_factory(self):
        r = Registry()
        called = []
        r.register("policy", "feedback", 1, lambda: called.append(True), ("wire-v2",))
        self.assertCode("duplicate_registry_id", r.register, "policy", "feedback", 1, lambda: None)
        self.assertCode("incompatible_version", r.create, "policy", "feedback", 2)
        self.assertCode("missing_capability", r.create, "policy", "feedback", require=("privileged",))
        self.assertCode("invalid_type", r.create, "policy", "feedback", True)
        self.assertEqual(called, [])
        r.create("policy", "feedback", require=("wire-v2",))
        self.assertEqual(called, [True])
        self.assertEqual(r.list("policy")[0].id, "feedback")


if __name__ == "__main__":
    unittest.main()
