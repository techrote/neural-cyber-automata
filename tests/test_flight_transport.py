import unittest

from nca.adapters.flight import (
    ACTION_PROFILE,
    ACTION_VERSION,
    ACTOR_ID,
    CONTROLLER_SHA256,
    COURSE_ID,
    COURSE_VERSION,
    DECISION_TICKS,
    EPISODE_VERSION,
    FLAT_CHANNELS,
    FLAT_OBSERVATION_VERSION,
    HOST_SCHEDULE,
    LEASE_TICKS,
    OBSERVATION_VERSION,
    PHYSICS_HZ,
    SPLIT_GEOMETRY_VERSION,
    TRAINER_PROFILE_SHA256,
    FlightProtocolError,
    UnknownConsumptionError,
)
from nca.adapters.flight_transport import (
    FLIGHT_PROTOCOL,
    FlightGodotTransport,
    FlightRpcTransport,
    inspect_registered_host,
)
from nca.adapters.protocol import ProtocolError


class Connection:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class Channel:
    def __init__(self, response=None, error=None, send_error=None):
        self.connection = Connection()
        self.response = response
        self.error = error
        self.send_error = send_error
        self.sent = []

    def send(self, value):
        self.sent.append(dict(value))
        if self.send_error is not None:
            raise self.send_error

    def receive(self):
        if self.error is not None:
            raise self.error
        if callable(self.response):
            return self.response(self.sent[-1])
        return self.response


class Accounting:
    def __init__(self):
        self.events = []
        self.next_id = 0

    def _reserve(self, name, **fields):
        self.next_id += 1
        reservation = f"r{self.next_id}"
        self.events.append(("reserve", name, reservation, fields))
        return reservation

    def reserve_reset(self, **fields):
        return self._reserve("reset", **fields)

    def reserve_step(self, **fields):
        return self._reserve("step", **fields)

    def reserve_cleanup(self, **fields):
        return self._reserve("cleanup", **fields)

    def reserve_host_call(self, **fields):
        return self._reserve("host_call", **fields)

    def complete(self, reservation, **fields):
        self.events.append(("complete", reservation, fields))

    def fail(self, reservation, reason, **fields):
        self.events.append(("failure", reservation, reason, fields))

    def mark_unknown(self, reservation, reason, **fields):
        self.events.append(("unknown", reservation, reason, fields))


def hello(**overrides):
    value = {
        "protocol": FLIGHT_PROTOCOL,
        "id": 0,
        "ok": True,
        "episode_version": EPISODE_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "flat_observation_version": FLAT_OBSERVATION_VERSION,
        "flat_observation_length": FLAT_CHANNELS,
        "action_version": ACTION_VERSION,
        "action_profile": ACTION_PROFILE,
        "course_id": COURSE_ID,
        "course_version": COURSE_VERSION,
        "split_geometry_version": SPLIT_GEOMETRY_VERSION,
        "host_schedule": HOST_SCHEDULE,
        "physics_hz": PHYSICS_HZ,
        "decision_ticks": DECISION_TICKS,
        "lease_ticks": LEASE_TICKS,
        "actor_id": ACTOR_ID,
        "producer": "policy",
        "trainer_profile_sha256": TRAINER_PROFILE_SHA256,
        "controller_sha256": CONTROLLER_SHA256,
        "native_coupling": True,
    }
    value.update(overrides)
    return value


def success(request, **fields):
    return {
        "protocol": FLIGHT_PROTOCOL,
        "id": request["id"],
        "ok": True,
        "consumption": "confirmed",
        **fields,
    }


class TransportTests(unittest.TestCase):
    def test_handshake_binds_split_geometry_v3(self):
        FlightRpcTransport(Channel(), hello=hello())
        with self.assertRaisesRegex(
            FlightProtocolError,
            "split_geometry_version",
        ):
            FlightRpcTransport(
                Channel(),
                hello=hello(split_geometry_version=2),
            )

    def test_matching_reply_preserves_sequence_and_consumption(self):
        channel = Channel(
            response=lambda request:
                success(request, closed=True)
        )
        transport = FlightRpcTransport(channel)
        reply = transport.request({
            "sequence": 1,
            "command": "close",
        })
        self.assertEqual(reply["sequence"], 1)
        self.assertEqual(
            reply["consumption"], "confirmed"
        )
        self.assertTrue(reply["closed"])
        self.assertEqual(
            channel.sent[0]["op"], "close"
        )

    def test_accounted_tuning_reset_reserves_tuning_episode_atomically(self):
        accounting = Accounting()
        transport = FlightRpcTransport(
            Channel(response=lambda request: success(
                request,
                transition={
                    "ticks_consumed": 0,
                    "failed": False,
                    "reason": "reset",
                },
            )),
            accounting=accounting,
        )
        transport.request({
            "sequence": 1,
            "command": "reset",
            "split": "tuning",
        })
        self.assertEqual(
            accounting.events[0],
            ("reserve", "reset", "r1", {"tuning": True}),
        )
        self.assertEqual(
            accounting.events[-1],
            ("complete", "r1", {"confirmed_physical_ticks": 0}),
        )

    def test_accounting_reserves_step_before_send_and_records_partial_ticks(self):
        accounting = Accounting()

        class OrderedChannel(Channel):
            def send(self, value):
                self.sent.append(dict(value))
                accounting.events.append(("send", value["op"]))
            def receive(self):
                accounting.events.append(("receive",))
                return success(
                    self.sent[-1],
                    transition={
                        "ticks_consumed": 3,
                        "failed": False,
                        "reason": "terminal",
                    },
                )

        transport = FlightRpcTransport(
            OrderedChannel(),
            accounting=accounting,
            capture_labels_per_step=1,
        )
        reply = transport.request({
            "sequence": 1,
            "command": "step",
            "repeat_ticks": 6,
        })
        self.assertEqual(reply["transition"]["ticks_consumed"], 3)
        self.assertEqual(accounting.events[0][0:2], ("reserve", "step"))
        self.assertEqual(
            accounting.events[0][3],
            {"maximum_ticks": 6, "capture_labels": 1},
        )
        self.assertEqual(accounting.events[1], ("send", "step"))
        self.assertEqual(accounting.events[2], ("receive",))
        self.assertEqual(
            accounting.events[3],
            ("complete", "r1", {"confirmed_physical_ticks": 3}),
        )

    def test_accounted_lost_reply_marks_reserved_work_unknown(self):
        accounting = Accounting()
        transport = FlightRpcTransport(
            Channel(error=ProtocolError("dropped")),
            accounting=accounting,
        )
        with self.assertRaises(UnknownConsumptionError):
            transport.request({
                "sequence": 1,
                "command": "step",
                "repeat_ticks": 6,
            })
        self.assertEqual(accounting.events[0][0:2], ("reserve", "step"))
        self.assertEqual(accounting.events[-1][0], "unknown")

    def test_write_failure_after_send_starts_is_unknown(self):
        channel = Channel(
            send_error=ProtocolError("short write")
        )
        transport = FlightRpcTransport(channel)
        with self.assertRaises(
            UnknownConsumptionError
        ) as raised:
            transport.request({
                "sequence": 1,
                "command": "reset",
            })
        self.assertIn(
            "send_started", raised.exception.reason
        )
        self.assertTrue(
            channel.connection.closed
        )

    def test_dropped_reply_after_dispatch_is_unknown_and_channel_closes(self):
        channel = Channel(
            error=ProtocolError("dropped")
        )
        transport = FlightRpcTransport(channel)
        with self.assertRaises(
            UnknownConsumptionError
        ) as raised:
            transport.request({
                "sequence": 1,
                "command": "step",
            })
        self.assertEqual(
            raised.exception.sequence, 1
        )
        self.assertTrue(
            channel.connection.closed
        )

    def test_duplicated_and_reordered_reply_ids_are_unknown(self):
        for reply_id in (0, 2):
            channel = Channel(response={
                "protocol": FLIGHT_PROTOCOL,
                "id": reply_id,
                "ok": True,
                "consumption": "confirmed",
            })
            transport = FlightRpcTransport(channel)
            with self.assertRaises(
                UnknownConsumptionError
            ):
                transport.request({
                    "sequence": 1,
                    "command": "step",
                })
            self.assertTrue(
                channel.connection.closed
            )

    def test_malformed_reply_and_missing_consumption_are_unknown(self):
        malformed = [
            {
                "protocol": FLIGHT_PROTOCOL,
                "id": 1,
                "consumption": "confirmed",
            },
            {
                "protocol": FLIGHT_PROTOCOL,
                "id": 1,
                "ok": True,
            },
            {
                "protocol": FLIGHT_PROTOCOL,
                "id": 1,
                "ok": True,
                "consumption": "maybe",
            },
        ]
        for response in malformed:
            channel = Channel(response=response)
            transport = FlightRpcTransport(channel)
            with self.assertRaises(
                UnknownConsumptionError
            ):
                transport.request({
                    "sequence": 1,
                    "command": "reset",
                })
            self.assertTrue(
                channel.connection.closed
            )

    def test_preconsumption_host_refusal_is_not_replayed(self):
        channel = Channel(response={
            "protocol": FLIGHT_PROTOCOL,
            "id": 1,
            "ok": False,
            "consumption": "none",
            "error": "stale_or_foreign_authority",
        })
        transport = FlightRpcTransport(channel)
        with self.assertRaises(
            FlightProtocolError
        ):
            transport.request({
                "sequence": 1,
                "command": "step",
            })
        self.assertTrue(
            channel.connection.closed
        )

    def test_success_cannot_claim_unconsumed_work(self):
        channel = Channel(response={
            "protocol": FLIGHT_PROTOCOL,
            "id": 1,
            "ok": True,
            "consumption": "none",
        })
        transport = FlightRpcTransport(channel)
        with self.assertRaises(
            FlightProtocolError
        ):
            transport.request({
                "sequence": 1,
                "command": "step",
            })
        self.assertTrue(
            channel.connection.closed
        )

    def test_actual_host_launcher_fails_before_inspecting_or_spawning_unaccepted_build(self):
        with self.assertRaisesRegex(
            FlightProtocolError,
            "actual-host conformance pending",
        ):
            inspect_registered_host(
                "definitely-not-a-checkout",
                "definitely-not-godot",
            )
        with self.assertRaisesRegex(
            FlightProtocolError,
            "actual-host conformance pending",
        ):
            FlightGodotTransport(
                "definitely-not-a-checkout",
                "definitely-not-godot",
            )


if __name__ == "__main__":
    unittest.main()
