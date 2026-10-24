from __future__ import annotations

import copy
import os
import unittest

from nca.adapters.protocol import HostError, ProtocolError
from nca.ik.host import (
    ACTIVE_JOINT_IDS, CAPABILITY, DESCRIPTOR_SHA256, DESCRIPTOR_SOURCE,
    JOINT_ACTIVE, JOINT_IDS, JOINT_PRESENT, PART_IDS, PART_PRESENT,
    PROFILE_ID, PROFILE_SHA256, TOPOLOGY_SHA256, GodotJointHost,
    JointTargetClient, ReceiptTracker, make_submit, validate_admission,
    validate_capability, validate_grant, validate_observation,
)


IDENTITY = {
    "host_revision": "e" * 40,
    "runtime_sha256": "f" * 64,
}
POLICY = "c" * 64


def capability(identity=IDENTITY):
    return {
        "protocol": "nca-host/1", "operation": "ik_capabilities", "version": 1,
        "capability": CAPABILITY, "mode": "exclusive_joint_targets",
        "host_source_commit": identity["host_revision"],
        "host_runtime_sha256": identity["runtime_sha256"],
        "profile_id": PROFILE_ID, "profile_sha256": PROFILE_SHA256,
        "descriptor_file_sha256": DESCRIPTOR_SHA256,
        "descriptor_source_commit": DESCRIPTOR_SOURCE,
        "topology_sha256": TOPOLOGY_SHA256,
        "effector": {"id": "right_sole", "part_id": "foot_1", "point_local_cells": [0.0, 1.0]},
        "part_ids": list(PART_IDS), "joint_ids": list(JOINT_IDS),
        "active_joint_ids": list(ACTIVE_JOINT_IDS),
        "part_capacity": 8, "joint_capacity": 7, "angle_unit": "radians",
        "length_unit": "game_cells", "target_mode": ["joint_position", "joint_delta"],
        "target_limits_radians": [[-0.72, 0.72], [-0.27, 0.27]],
        "max_delta_radians": 0.25, "base_hz": 60, "decision_ticks": 6,
        "max_observation_age_ticks": 12, "max_lease_ticks": 18,
        "helper_lease_ceiling_ticks": 12, "deadline_exclusive": True,
        "source_grant_required": True, "non_commanded": "passive_constraints",
        "wire_buttons_mounts": "disabled_disarmed",
        "authored_gyro": {"neutral_target_radians": 0.0, "gain_grounded": 1.0,
                         "gain_airborne": 0.65, "gyro": 26.0, "gyro_kp": 10.0,
                         "damping": 3.0, "play_radians": 0.055,
                         "grounded_slope_lean": True, "button_boost": False},
        "motor": {"effort": 1.0, "speed_cap_radians_per_second": 7.0,
                  "request_slew_radians_per_second_squared": 20.0,
                  "pin_joint_max_force_parameter": 100.0,
                  "max_force_is_measured_torque": False},
        "force_telemetry": "unavailable", "passive_force_decomposition": "unavailable",
        "receipts": ["requested", "admitted", "limited", "motor", "motion", "integrity"],
        "packet_kind": "ik_submit", "observation_schema": "ik_host_observation/v1",
        "actor_id": 0, "run_id": "run-1", "epoch": 1, "incarnation": 1,
        "control_generation": 1, "source_owner": "policy",
    }


def grant(cap=None, *, source="numerical-reference", generation=2):
    cap = capability() if cap is None else cap
    return {
        "protocol": "nca-host/1", "operation": "ik_grant", "version": 1,
        "run_id": cap["run_id"], "actor_id": cap["actor_id"], "epoch": cap["epoch"],
        "incarnation": cap["incarnation"], "control_generation": generation,
        "source_id": source, "policy_sha256": POLICY, "goal_id": "reach-1",
        "goal_deadline_tick": 40, "grant_id": "grant-1",
        "grant_generation": generation, "producer": "policy",
        "profile_sha256": PROFILE_SHA256, "topology_sha256": TOPOLOGY_SHA256,
        "joint_ids": list(ACTIVE_JOINT_IDS),
    }


def observation(g=None, *, tick=0):
    g = grant() if g is None else g
    return {
        "protocol": "nca-host/1", "operation": "ik_observe",
        "schema": "ik_host_observation/v1", "version": 1,
        "run_id": g["run_id"], "actor_id": g["actor_id"], "epoch": g["epoch"],
        "incarnation": g["incarnation"], "control_generation": g["control_generation"],
        "grant_id": g["grant_id"], "source_id": g["source_id"], "goal_id": g["goal_id"],
        "observed_tick": tick, "observation_id": f"observation-{tick}",
        "observation_sha256": "a" * 64, "profile_sha256": PROFILE_SHA256,
        "topology_sha256": TOPOLOGY_SHA256,
        "root_pose": {"position_cells": [400.0, 400.0], "angle_radians": 0.0},
        "part_ids": list(PART_IDS), "joint_ids": list(JOINT_IDS),
        "part_present": list(PART_PRESENT), "joint_present": list(JOINT_PRESENT),
        "joint_measured": list(JOINT_PRESENT), "joint_active": list(JOINT_ACTIVE),
        "joint_positions_radians": [0.0, 0.0, 0.1, -0.05, None, None, None],
        "measured_chain_radians": [0.1, -0.05], "evidence_kind": "host_observation",
    }


def admission(packet):
    return {
        "protocol": "nca-host/1", "operation": "ik_submit", "accepted": True,
        "requested": copy.deepcopy(packet),
        "admitted": {"owner_tick": packet["observed_tick"],
                     "references_radians": list(packet["joint_values"]),
                     "mode": "exclusive_joint_targets",
                     "joint_ids": list(ACTIVE_JOINT_IDS),
                     "same_owner_and_limits_for_producers": True},
        "limited": {"reference_projection": False, "reference_interpolation": False,
                    "motor_speed_and_slew_limited_at_consumption": True},
        "integrity": {"original_observed_tick": packet["observed_tick"],
                      "exclusive_deadline_tick": packet["observed_tick"] + packet["lease_ticks"],
                      "applied_tick_count": 0, "terminal_boundary": "not_consumed",
                      "missing_receipt_count": 0, "unknown_receipt_count": 0},
    }


def receipt(packet, tick):
    return {
        "protocol": "nca-host/1", "operation": "ik_receipt", "version": 1,
        "accepted": True, "requested": copy.deepcopy(packet),
        "admitted": {"owner_tick": tick, "sequence": packet["sequence"],
                     "control_generation": packet["control_generation"],
                     "grant_id": packet["grant_id"], "references_radians": list(packet["joint_values"]),
                     "mode": "exclusive_joint_targets"},
        "limited": {"reference_radians": list(packet["joint_values"]),
                    "motor_slew_and_deadzone": []},
        "motor": {"commanded_joint_ids": list(ACTIVE_JOINT_IDS),
                  "passive_joint_ids": ["hip_-1", "ankle_-1"], "motors": []},
        "motion": {"available": True, "completed_tick": tick + 1,
                   "root_position": [400.0, 400.0], "root_angle": 0.0,
                   "joint_angles_radians": [0.1, -0.05],
                   "tracking_error_radians": [0.1, -0.1],
                   "part_motion": [{}, {}, {}, {}, {}],
                   "joint_motion": [{}, {}, {}, {}],
                   "grounded": True, "contact_normal": [0.0, -1.0],
                   "per_joint_force_telemetry": None,
                   "passive_force_decomposition": None},
        "integrity": {"requested_observed_tick": packet["observed_tick"],
                      "consumed_base_tick": tick, "expected_physics_step": tick + 1,
                      "actual_physics_step": tick + 1, "completed_base_tick": tick + 1,
                      "terminal_boundary": "completed", "missing_receipt_count": 0,
                      "unknown_receipt_count": 0},
    }


def terminal(packet, reason="owner_cancelled"):
    return {
        "protocol": "nca-host/1", "operation": "ik_terminal", "version": 1,
        "accepted": False, "requested": copy.deepcopy(packet),
        "admitted": {"owner_tick": 3, "no_new_actuation": True},
        "limited": {"motor_disarmed": True},
        "motor": {"mode": "passive_no_grant", "commanded_joint_ids": [],
                  "passive_joint_ids": list(JOINT_IDS)},
        "motion": {"available": False, "reason": "no_physics_consumption"},
        "integrity": {"requested_observed_tick": packet["observed_tick"],
                      "consumed_base_tick": None, "completed_base_tick": None,
                      "actual_physics_step": None, "terminal_boundary": reason,
                      "cancel_confirmed": True, "missing_receipt_count": 0,
                      "unknown_receipt_count": 0},
    }


class FakeTransport:
    def __init__(self):
        self.cap = capability()
        self.current_grant = None
        self.current_observation = None
        self.rows = {}
        self.tick = 0
        self.fail_submit = False

    def request(self, operation, **fields):
        if operation == "ik_capabilities":
            return {"capability": copy.deepcopy(self.cap)}
        if operation == "ik_grant":
            self.current_grant = grant(self.cap, source=fields["source_id"],
                                       generation=self.cap["control_generation"] + 1)
            self.current_grant["policy_sha256"] = fields["policy_sha256"]
            self.current_grant["goal_id"] = fields["goal_id"]
            self.current_grant["goal_deadline_tick"] = fields["goal_deadline_tick"]
            return {"grant": copy.deepcopy(self.current_grant)}
        if operation == "ik_observe":
            self.current_observation = observation(self.current_grant, tick=self.tick)
            return {"observation": copy.deepcopy(self.current_observation)}
        if operation == "ik_submit":
            if self.fail_submit:
                raise ProtocolError("lost reply")
            packet = copy.deepcopy(fields["packet"])
            self.rows[packet["sequence"]] = [admission(packet)]
            return {"admission": admission(packet)}
        if operation == "advance":
            for sequence, rows in self.rows.items():
                packet = rows[0]["requested"]
                for _ in range(fields["ticks"]):
                    rows.append(receipt(packet, self.tick))
                    self.tick += 1
                    break
                break
            return {"ticks": fields["ticks"], "observation": {}}
        if operation == "ik_receipts":
            return {"receipts": copy.deepcopy(self.rows[fields["sequence"]])}
        if operation == "ik_cancel":
            value = {"protocol": "nca-host/1", "operation": "ik_cancel",
                     "confirmed": True, "tick": self.tick, "run_id": "run-1",
                     "actor_id": 0, "epoch": 1, "revoked_generation": 2,
                     "new_generation": 3, "grant_id": fields["grant_id"],
                     "old_motor_references_revoked": True, "momentum_reset": False}
            return {"cancellation": value}
        raise AssertionError(operation)


class JointHostSchemaTests(unittest.TestCase):
    def test_valid_capability(self):
        self.assertEqual(validate_capability(capability(), IDENTITY)["capability"], CAPABILITY)

    def test_capability_missing_field(self):
        value = capability()
        value.pop("motor")
        with self.assertRaises(ProtocolError):
            validate_capability(value, IDENTITY)

    def test_capability_extra_field(self):
        value = capability()
        value["reward"] = 1
        with self.assertRaises(ProtocolError):
            validate_capability(value, IDENTITY)

    def test_capability_source_identity(self):
        value = capability()
        value["host_source_commit"] = "d" * 40
        with self.assertRaises(ProtocolError):
            validate_capability(value, IDENTITY)

    def test_capability_runtime_identity(self):
        value = capability()
        value["host_runtime_sha256"] = "d" * 64
        with self.assertRaises(ProtocolError):
            validate_capability(value, IDENTITY)

    def test_capability_profile(self):
        value = capability()
        value["profile_sha256"] = "d" * 64
        with self.assertRaises(ProtocolError):
            validate_capability(value, IDENTITY)

    def test_capability_topology(self):
        value = capability()
        value["topology_sha256"] = "d" * 64
        with self.assertRaises(ProtocolError):
            validate_capability(value, IDENTITY)

    def test_capability_joint_order(self):
        value = capability()
        value["active_joint_ids"].reverse()
        with self.assertRaises(ProtocolError):
            validate_capability(value, IDENTITY)

    def test_capability_limits(self):
        value = capability()
        value["target_limits_radians"][0][1] = 0.73
        with self.assertRaises(ProtocolError):
            validate_capability(value, IDENTITY)

    def test_capability_assistance(self):
        value = capability()
        value["authored_gyro"]["gain_airborne"] = 1.0
        with self.assertRaises(ProtocolError):
            validate_capability(value, IDENTITY)

    def test_capability_force_truth(self):
        value = capability()
        value["motor"]["max_force_is_measured_torque"] = True
        with self.assertRaises(ProtocolError):
            validate_capability(value, IDENTITY)

    def test_grant_must_advance_generation(self):
        cap = validate_capability(capability(), IDENTITY)
        with self.assertRaises(ProtocolError):
            validate_grant(grant(cap, generation=1), cap)

    def test_grant_binds_profile(self):
        cap = validate_capability(capability(), IDENTITY)
        value = grant(cap)
        value["topology_sha256"] = "d" * 64
        with self.assertRaises(ProtocolError):
            validate_grant(value, cap)

    def test_observation_masks(self):
        g = grant()
        value = observation(g)
        self.assertEqual(validate_observation(value, g)["measured_chain_radians"], [0.1, -0.05])
        value["joint_active"][0] = True
        with self.assertRaises(ProtocolError):
            validate_observation(value, g)

    def test_observation_chain_matches_slots(self):
        g = grant()
        value = observation(g)
        value["measured_chain_radians"][1] = -0.04
        with self.assertRaises(ProtocolError):
            validate_observation(value, g)

    def test_position_packet(self):
        g, view = grant(), observation()
        packet = make_submit(g, view, 0, [0.2, -0.1])
        self.assertEqual(packet["joint_ids"], list(ACTIVE_JOINT_IDS))
        self.assertEqual(packet["observation_id"], view["observation_id"])

    def test_delta_packet(self):
        packet = make_submit(grant(), observation(), 0, [0.25, -0.25], mode="joint_delta")
        self.assertEqual(packet["target_mode"], "joint_delta")

    def test_helper_lease_is_twelve(self):
        with self.assertRaises(ProtocolError):
            make_submit(grant(), observation(), 0, [0.0, 0.0], lease_ticks=13)

    def test_position_bounds(self):
        with self.assertRaises(ProtocolError):
            make_submit(grant(), observation(), 0, [0.721, 0.0])

    def test_delta_bounds(self):
        with self.assertRaises(ProtocolError):
            make_submit(grant(), observation(), 0, [0.251, 0.0], mode="joint_delta")

    def test_boolean_is_not_number(self):
        with self.assertRaises(ProtocolError):
            make_submit(grant(), observation(), 0, [True, 0.0])

    def test_admission_exact_request(self):
        packet = make_submit(grant(), observation(), 0, [0.2, -0.1])
        value = admission(packet)
        self.assertTrue(validate_admission(value, packet)["accepted"])
        value["requested"]["sequence"] = 9
        with self.assertRaises(ProtocolError):
            validate_admission(value, packet)

    def test_receipt_continuity(self):
        packet = make_submit(grant(), observation(), 0, [0.2, -0.1])
        tracker = ReceiptTracker(packet)
        rows = tracker.consume([admission(packet), receipt(packet, 0), receipt(packet, 1)])
        self.assertEqual(len(rows), 3)
        self.assertEqual(tracker.last_completed_tick, 2)

    def test_receipt_gap_refused(self):
        packet = make_submit(grant(), observation(), 0, [0.2, -0.1])
        tracker = ReceiptTracker(packet)
        with self.assertRaises(ProtocolError):
            tracker.consume([admission(packet), receipt(packet, 1)])

    def test_receipt_missing_marker_refused(self):
        packet = make_submit(grant(), observation(), 0, [0.2, -0.1])
        row = receipt(packet, 0)
        row["integrity"]["missing_receipt_count"] = 1
        with self.assertRaises(ProtocolError):
            ReceiptTracker(packet).consume([row])

    def test_receipt_force_fabrication_refused(self):
        packet = make_submit(grant(), observation(), 0, [0.2, -0.1])
        row = receipt(packet, 0)
        row["motion"]["per_joint_force_telemetry"] = 3.0
        with self.assertRaises(ProtocolError):
            ReceiptTracker(packet).consume([row])

    def test_terminal_must_be_confirmed(self):
        packet = make_submit(grant(), observation(), 0, [0.2, -0.1])
        tracker = ReceiptTracker(packet)
        tracker.consume([terminal(packet)])
        self.assertEqual(tracker.terminal, "owner_cancelled")
        bad = terminal(packet)
        bad["integrity"]["cancel_confirmed"] = False
        with self.assertRaises(ProtocolError):
            ReceiptTracker(packet).consume([bad])


class JointTargetClientTests(unittest.TestCase):
    def setUp(self):
        self.transport = FakeTransport()
        self.client = JointTargetClient(self.transport, identity=IDENTITY)

    def test_negotiation(self):
        self.assertEqual(self.client.negotiate()["host_runtime_sha256"], IDENTITY["runtime_sha256"])

    def test_grant_observe_submit(self):
        self.client.negotiate()
        self.client.grant("numerical-reference", POLICY, "reach-1", 40)
        view = self.client.observe()
        result = self.client.submit([0.2, -0.1], observation=view)
        self.assertTrue(result["accepted"])
        self.assertEqual(self.client.next_sequence, 1)

    def test_numerical_and_learned_sources_share_packet_limits(self):
        packets = []
        for source in ("numerical-reference", "learned-model"):
            transport = FakeTransport()
            client = JointTargetClient(transport, identity=IDENTITY)
            client.negotiate()
            client.grant(source, POLICY, "reach-1", 40)
            result = client.submit([0.2, -0.1], observation=client.observe())
            packets.append(result["requested"])
        for key in ("joint_ids", "joint_values", "target_mode", "lease_ticks",
                    "profile_sha256", "topology_sha256"):
            self.assertEqual(packets[0][key], packets[1][key])
        self.assertNotEqual(packets[0]["source_id"], packets[1]["source_id"])

    def test_receipt_tracker_through_client(self):
        self.client.negotiate()
        self.client.grant("numerical-reference", POLICY, "reach-1", 40)
        self.client.submit([0.2, -0.1], observation=self.client.observe())
        self.client.advance(1)
        rows = self.client.receipts(0)
        self.assertEqual(rows[-1]["integrity"]["completed_base_tick"], 1)

    def test_lost_submit_requires_confirmed_cancel(self):
        self.client.negotiate()
        self.client.grant("numerical-reference", POLICY, "reach-1", 40)
        self.client.observe()
        self.transport.fail_submit = True
        with self.assertRaises(ProtocolError):
            self.client.submit([0.2, -0.1])
        with self.assertRaises(HostError):
            self.client.submit([0.2, -0.1])
        result = self.client.cancel()
        self.assertTrue(result["confirmed"])
        self.assertFalse(self.client.uncertain)

    def test_cancel_cannot_claim_momentum_reset(self):
        self.client.negotiate()
        self.client.grant("numerical-reference", POLICY, "reach-1", 40)
        original = self.transport.request

        def bad(operation, **fields):
            result = original(operation, **fields)
            if operation == "ik_cancel":
                result["cancellation"]["momentum_reset"] = True
            return result

        self.transport.request = bad
        with self.assertRaises(ProtocolError):
            self.client.cancel()


@unittest.skipUnless(os.environ.get("NCA_IK_HOST_CHECKOUT") and os.environ.get("NCA_IK_GODOT"),
                     "actual joint-target host is supplied only by pinned host conformance")
class ActualJointHostTests(unittest.TestCase):
    def test_actual_host_authority_receipts_contact_expiry_and_takeover(self):
        checkout = os.environ["NCA_IK_HOST_CHECKOUT"]
        godot = os.environ["NCA_IK_GODOT"]
        with GodotJointHost(checkout, godot, timeout=60.0) as host:
            client = JointTargetClient(host, identity=host.identity)
            cap = client.negotiate()
            self.assertEqual(cap["host_source_commit"], host.identity["host_revision"])
            self.assertEqual(cap["host_runtime_sha256"], host.identity["runtime_sha256"])
            self.assertEqual(cap["active_joint_ids"], list(ACTIVE_JOINT_IDS))

            first = client.grant("numerical-reference", POLICY, "reach-actual", 40)
            view = client.observe()
            admitted = client.submit([0.55, 0.20], observation=view)
            packet = admitted["requested"]

            # A foreign incarnation may be refused, but cannot revoke the real owner.
            foreign = copy.deepcopy(packet)
            foreign["incarnation"] += 1
            refusal = host.request("ik_submit", packet=foreign)["admission"]
            self.assertFalse(refusal["accepted"])
            duplicate = host.request("ik_submit", packet=packet)["admission"]
            self.assertTrue(duplicate["accepted"])
            self.assertTrue(duplicate["duplicate_lookup"])

            client.advance(6)
            rows = client.receipts(0)
            consumed = [row for row in rows if row["operation"] == "ik_receipt"]
            self.assertEqual(len(consumed), 6)
            self.assertTrue(all(row["motion"]["available"] for row in consumed))
            self.assertTrue(any(row["motion"]["grounded"] for row in consumed))
            self.assertTrue(all(row["motion"]["per_joint_force_telemetry"] is None for row in consumed))
            self.assertTrue(all(row["motion"]["passive_force_decomposition"] is None for row in consumed))
            self.assertTrue(all(row["motor"]["commanded_joint_ids"] == list(ACTIVE_JOINT_IDS)
                                for row in consumed))
            # The static floor/side-stop constrains the whole actor while the owner
            # still reports a finite tracking residual rather than teleporting it.
            self.assertTrue(any(max(abs(float(v)) for v in row["motion"]["tracking_error_radians"]) > 1e-4
                                for row in consumed))

            before_cancel = host.request("advance", ticks=1)["observation"]
            cancelled = client.cancel()
            self.assertFalse(cancelled["momentum_reset"])
            after_cancel = host.request("advance", ticks=1)["observation"]
            self.assertEqual(before_cancel["position"], after_cancel["position"],
                             "cancel must not teleport the root")
            old = host.request("ik_submit", packet=packet)["admission"]
            self.assertFalse(old["accepted"])

            # Source takeover creates a new generation.  Numerical and learned
            # controllers still see the identical negotiated physical limits.
            cap2 = client.negotiate()
            second = client.grant("learned-model", POLICY, "reach-learned", 60)
            self.assertGreater(second["control_generation"], first["control_generation"])
            self.assertEqual(cap2["motor"], cap["motor"])
            view2 = client.observe()
            client.submit([0.55, 0.20], observation=view2)
            client.advance(12)
            final = client.receipts(0)
            consumed2 = [row for row in final if row["operation"] == "ik_receipt"]
            self.assertEqual(len(consumed2), 12)
            client.advance(1)
            expired = client.receipts(0)
            self.assertEqual(expired[-1]["operation"], "ik_terminal")
            self.assertTrue(expired[-1]["integrity"]["cancel_confirmed"])


if __name__ == "__main__":
    unittest.main()
