import unittest

from nca.adapters.flight import (
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


def success(request, **fields):
    return {
        "protocol": FLIGHT_PROTOCOL,
        "id": request["id"],
        "ok": True,
        "consumption": "confirmed",
        **fields,
    }


class TransportTests(unittest.TestCase):
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
