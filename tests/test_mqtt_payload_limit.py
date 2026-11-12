"""Oversized publishes: refused before the broker can drop them silently.

Cyberwave's brokers drop any payload over 32 MiB (from CYB-4375 on). Over MQTT
3.1.1 the publisher cannot tell: a QoS 0 message vanishes and a QoS 1 PUBACK
reports success. Over MQTT 5 the PUBACK does say "Packet too large", but paho
2.1 cannot parse that code on a PUBACK, and the exception kills its network
thread -- the client stops sending, receiving and reconnecting, with no
``on_disconnect`` to report it. These tests pin both halves: the SDK refuses
what the broker would drop, and a refusal that still arrives is survivable.
"""

from __future__ import annotations

import logging
import struct
import threading
from unittest.mock import MagicMock, patch

import paho.mqtt.client as mqtt
import pytest
from paho.mqtt.enums import CallbackAPIVersion
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.reasoncodes import ReasonCode

from cyberwave import _paho_compat, _payload_limit
from cyberwave.config import CyberwaveConfig
from cyberwave.mqtt import (
    DEFAULT_MAX_PAYLOAD_BYTES,
    MAX_PAYLOAD_BYTES_ENV,
    CyberwaveMQTTClient,
)
from cyberwave.mqtt_client import CyberwaveMQTTClient as WrapperMQTTClient

TOPIC = "cyberwave/twin/8f1c2d3e-0000-4000-8000-000000000001/pointcloud"


def _client(**kwargs) -> CyberwaveMQTTClient:
    client = CyberwaveMQTTClient(
        mqtt_broker="localhost",
        mqtt_port=1883,
        mqtt_username="user",
        api_key="api_key_secret",
        auto_connect=False,
        **kwargs,
    )
    client.connected = True
    return client


def _accepting(mqtt_client_cls: MagicMock) -> MagicMock:
    mqtt_client = mqtt_client_cls.return_value
    mqtt_client.publish.return_value = MagicMock(rc=mqtt.MQTT_ERR_SUCCESS)
    return mqtt_client


def _connack(**values) -> MagicMock:
    properties = MagicMock()
    properties.MaximumPacketSize = values.get("MaximumPacketSize", 0)
    return properties


# -- the limit -------------------------------------------------------------


def test_the_default_limit_is_the_cyberwave_brokers_32_mib(monkeypatch) -> None:
    monkeypatch.delenv(MAX_PAYLOAD_BYTES_ENV, raising=False)
    with patch("cyberwave.mqtt.mqtt.Client"):
        assert _client().max_payload_bytes == DEFAULT_MAX_PAYLOAD_BYTES == 33554432


def test_an_oversized_payload_is_refused_before_paho_sees_it(caplog) -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = _accepting(mqtt_client_cls)
        client = _client(max_payload_bytes=1024)

        with caplog.at_level(logging.ERROR, logger="cyberwave.mqtt"):
            accepted = client.publish(TOPIC, {"data": "A" * 2048}, qos=1)

        assert accepted is False
        mqtt_client.publish.assert_not_called()
        assert f"Not publishing to {TOPIC}" in caplog.text
        assert "over the 1024-byte limit" in caplog.text


def test_a_payload_exactly_at_the_limit_is_sent() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = _accepting(mqtt_client_cls)
        client = _client(max_payload_bytes=64)

        assert client.publish(TOPIC, "A" * 64) is True
        assert client.publish(TOPIC, "A" * 65) is False
        assert mqtt_client.publish.call_count == 1


def test_the_limit_counts_the_serialized_payload_session_id_included() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = _accepting(mqtt_client_cls)
        client = _client(max_payload_bytes=40)

        # {"x": 1} alone is 8 bytes; ``publish`` adds the session id before it
        # measures, and that is what the broker receives.
        assert client.publish(TOPIC, {"x": 1}) is False
        mqtt_client.publish.assert_not_called()


def test_text_is_measured_in_utf8_bytes_not_characters() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        _accepting(mqtt_client_cls)
        client = _client(max_payload_bytes=10)

        assert client.publish(TOPIC, "é" * 5) is True  # 10 bytes
        assert client.publish(TOPIC, "é" * 6) is False  # 12 bytes, 6 chars


def test_binary_payloads_are_measured_too() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        _accepting(mqtt_client_cls)
        client = _client(max_payload_bytes=10)

        assert client.publish(TOPIC, b"\x00" * 10) is True
        assert client.publish(TOPIC, bytearray(11)) is False
        assert client.publish(TOPIC, memoryview(b"\x00" * 11)) is False


def test_the_sensor_publishers_report_a_refusal() -> None:
    # The two publishers most likely to go over the limit, so the two whose
    # callers most need to hear it, e.g. to downsample and send again.
    np = pytest.importorskip("numpy")
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        _accepting(mqtt_client_cls)
        client = _client(max_payload_bytes=1024)
        uuid = "8f1c2d3e-0000-4000-8000-000000000001"

        assert client.publish_pointcloud(uuid, np.zeros((4, 6))) is True
        assert client.publish_pointcloud(uuid, np.zeros((1024, 6))) is False
        assert client.publish_depth_frame(uuid, {"data": ""}) is True
        assert client.publish_depth_frame(uuid, {"data": "A" * 1024}) is False


def test_a_cloud_too_large_to_record_is_sent_with_a_throttled_warning(caplog) -> None:
    # Vector's size_guard_colored_pointcloud drops a pointcloud payload over
    # 8 MiB unread. The broker still delivers it to live viewers, so it is
    # sent, but the driver hears that it was not recorded.
    limit = _payload_limit.POINTCLOUD_INGEST_MAX_PAYLOAD_BYTES
    assert limit == 8 * 1024 * 1024
    with (
        patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls,
        patch("cyberwave.mqtt.time.monotonic") as monotonic,
    ):
        mqtt_client = _accepting(mqtt_client_cls)
        client = _client()
        monotonic.return_value = 100.0

        with caplog.at_level(logging.WARNING, logger="cyberwave.mqtt"):
            assert client.publish(TOPIC, b"\x00" * limit) is True
            assert client.publish(TOPIC, b"\x00" * (limit + 1)) is True
            assert client.publish(TOPIC, b"\x00" * (limit + 1)) is True

        assert mqtt_client.publish.call_count == 3
        warnings = [r for r in caplog.records if "not recorded" in r.getMessage()]
        assert len(warnings) == 1
        assert warnings[0].levelno == logging.WARNING
        assert f"{limit + 1} bytes" in warnings[0].getMessage()


def test_only_point_clouds_have_the_smaller_recording_limit() -> None:
    size = _payload_limit.POINTCLOUD_INGEST_MAX_PAYLOAD_BYTES + 1
    assert _payload_limit.unrecorded_reason(TOPIC, b"\x00" * size) is not None
    depth = TOPIC.replace("/pointcloud", "/depth")
    assert _payload_limit.unrecorded_reason(depth, b"\x00" * size) is None
    assert _payload_limit.unrecorded_reason(TOPIC, b"\x00" * (size - 1)) is None


def test_zero_turns_the_check_off() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = _accepting(mqtt_client_cls)
        client = _client(max_payload_bytes=0)

        assert client.publish(TOPIC, b"\x00" * (DEFAULT_MAX_PAYLOAD_BYTES + 1)) is True
        mqtt_client.publish.assert_called_once()


def test_the_environment_sets_the_limit_and_the_argument_wins(monkeypatch) -> None:
    monkeypatch.setenv(MAX_PAYLOAD_BYTES_ENV, "4096")
    with patch("cyberwave.mqtt.mqtt.Client"):
        assert _client().max_payload_bytes == 4096
        assert _client(max_payload_bytes=128).max_payload_bytes == 128


def test_the_wrapper_client_honours_the_environment(monkeypatch) -> None:
    # ``Cyberwave.mqtt`` hands out the wrapper, which does not forward a
    # ``max_payload_bytes`` of its own; the variable is how its users set it.
    monkeypatch.setenv(MAX_PAYLOAD_BYTES_ENV, "2048")
    config = CyberwaveConfig(api_key="api_key_secret", mqtt_username="user")
    with patch("cyberwave.mqtt.mqtt.Client"):
        wrapper = WrapperMQTTClient(config=config)
    assert wrapper._client.max_payload_bytes == 2048


@pytest.mark.parametrize(
    "raw",
    [
        "32MiB",
        "-1",
        "1.5",
        "+1",
        "1_000",
        "\u0661\u0662",
        "0\v",
        "\u00a00",
        "18446744073709551616",
        "99999999999999999999999",
        pytest.param("9" * 5000, id="over-cpythons-4300-digit-limit"),
        pytest.param(" \t", id="whitespace-only"),
    ],
)
def test_a_malformed_environment_value_warns_and_keeps_the_default(
    monkeypatch, caplog, raw
) -> None:
    # What the C++ SDK does with the same variable, value for value: int()
    # would take "+1" (a limit that refuses everything), "1_000" and
    # Arabic-Indic digits, and std::from_chars takes none of them. str.strip()
    # would reduce "0\v" and a no-break space before "0" to "0", no limit, and
    # int() has no SIZE_MAX to overflow.
    monkeypatch.setenv(MAX_PAYLOAD_BYTES_ENV, raw)
    with (
        patch("cyberwave.mqtt.mqtt.Client"),
        caplog.at_level(logging.WARNING, logger="cyberwave.mqtt"),
    ):
        assert _client().max_payload_bytes == DEFAULT_MAX_PAYLOAD_BYTES

    assert f"Invalid {MAX_PAYLOAD_BYTES_ENV}={raw!r}" in caplog.text


def test_leading_zeros_do_not_count_against_the_size_max_digits(monkeypatch) -> None:
    # std::from_chars reads this as 1. int() would refuse it outright, being
    # over CPython's 4300-digit conversion limit, so it must not see it whole.
    monkeypatch.setenv(MAX_PAYLOAD_BYTES_ENV, "0" * 5000 + "1")
    with patch("cyberwave.mqtt.mqtt.Client"):
        assert _client().max_payload_bytes == 1


def test_surrounding_whitespace_in_the_environment_is_ignored(monkeypatch) -> None:
    monkeypatch.setenv(MAX_PAYLOAD_BYTES_ENV, " 4096\n")
    with patch("cyberwave.mqtt.mqtt.Client"):
        assert _client().max_payload_bytes == 4096


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_a_malformed_argument_fails_loudly(value) -> None:
    with patch("cyberwave.mqtt.mqtt.Client"), pytest.raises(ValueError):
        _client(max_payload_bytes=value)


# -- the broker's own limit (MQTT 5) ---------------------------------------


def test_a_v5_brokers_maximum_packet_size_applies_as_well() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = _accepting(mqtt_client_cls)
        client = _client(protocol=mqtt.MQTTv5)
        client._on_connect(mqtt_client, None, {}, 0, _connack(MaximumPacketSize=256))

        # A PUBLISH spends at most 16 bytes beyond its topic and payload, so
        # this is the most payload a 256-byte packet can carry.
        fits = 256 - len(TOPIC) - 16
        assert client.publish(TOPIC, "A" * fits) is True
        assert client.publish(TOPIC, "A" * (fits + 1)) is False
        assert mqtt_client.publish.call_count == 1


def test_the_smaller_of_the_two_limits_wins() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = _accepting(mqtt_client_cls)
        client = _client(protocol=mqtt.MQTTv5, max_payload_bytes=100)
        client._on_connect(mqtt_client, None, {}, 0, _connack(MaximumPacketSize=4096))

        assert client.publish(TOPIC, "A" * 101) is False


def test_a_connack_without_a_maximum_packet_size_clears_the_last_one() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = _accepting(mqtt_client_cls)
        client = _client(protocol=mqtt.MQTTv5)
        client._on_connect(mqtt_client, None, {}, 0, _connack(MaximumPacketSize=256))
        client._on_connect(mqtt_client, None, {}, 0, _connack())

        assert client.publish(TOPIC, "A" * 4096) is True


def test_a_disconnect_forgets_the_brokers_maximum_packet_size() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = _accepting(mqtt_client_cls)
        client = _client(protocol=mqtt.MQTTv5)
        client._on_connect(mqtt_client, None, {}, 0, _connack(MaximumPacketSize=256))
        client._on_disconnect(mqtt_client, None, 0)

        # Negotiated per connection: a publish made before the next CONNACK
        # is held to the local limit only.
        client.connected = True
        assert client.publish(TOPIC, "A" * 4096) is True


def test_a_v311_client_ignores_connack_properties() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = _accepting(mqtt_client_cls)
        client = _client(protocol=mqtt.MQTTv311)
        client._on_connect(mqtt_client, None, {}, 0, _connack(MaximumPacketSize=256))

        assert client.publish(TOPIC, "A" * 4096) is True


# -- reporting -------------------------------------------------------------


def test_repeated_refusals_are_reported_once_per_interval(caplog) -> None:
    with (
        patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls,
        patch("cyberwave.mqtt.time.monotonic") as monotonic,
    ):
        _accepting(mqtt_client_cls)
        client = _client(max_payload_bytes=8)

        with caplog.at_level(logging.ERROR, logger="cyberwave.mqtt"):
            monotonic.return_value = 100.0
            client.publish(TOPIC, "A" * 9)
            monotonic.return_value = 101.0
            client.publish(TOPIC, "A" * 9)
            client.publish(TOPIC, "A" * 9)
            monotonic.return_value = 111.0
            client.publish(TOPIC, "A" * 9)

        refusals = [
            r.getMessage() for r in caplog.records if "Not publishing" in r.getMessage()
        ]
        assert len(refusals) == 2
        assert "(2 more since the last report)" in refusals[1]


def test_a_broker_refusal_on_puback_is_logged(caplog) -> None:
    client = _client(protocol=mqtt.MQTTv5)
    mid = client.client.publish("a/b", b"x", qos=1).mid
    refused = ReasonCode(PacketTypes.PUBACK)
    refused.unpack(bytes([_paho_compat.PACKET_TOO_LARGE]))

    with caplog.at_level(logging.ERROR, logger="cyberwave.mqtt"):
        client._on_publish(None, None, mid, ReasonCode(PacketTypes.PUBACK), None)
        assert caplog.text == ""
        client._on_publish(None, None, mid, refused, None)
        # A mid paho no longer holds is still reported, without a topic.
        client._on_publish(None, None, mid + 100, refused, None)

    assert (
        f"MQTT broker refused a publish to a/b (mid={mid}): Packet too large (149)"
        in caplog.text
    )
    assert (
        f"MQTT broker refused a publish (mid={mid + 100}): Packet too large (149)"
        in caplog.text
    )


def test_broker_refusals_are_throttled_per_topic(caplog) -> None:
    client = _client(protocol=mqtt.MQTTv5)
    refused = ReasonCode(PacketTypes.PUBACK)
    refused.unpack(bytes([_paho_compat.PACKET_TOO_LARGE]))
    mids = [client.client.publish(t, b"x", qos=1).mid for t in ("a/b", "c/d")]

    with caplog.at_level(logging.ERROR, logger="cyberwave.mqtt"):
        for mid in mids:
            client._on_publish(None, None, mid, refused, None)

    assert "refused a publish to a/b" in caplog.text
    assert "refused a publish to c/d" in caplog.text


# -- paho 2.1 and reason code 149 ------------------------------------------


@pytest.mark.parametrize("packet", [PacketTypes.PUBACK, PacketTypes.PUBREC])
def test_paho_can_parse_packet_too_large_on_a_publish_ack(packet) -> None:
    code = ReasonCode(packet)
    code.unpack(bytes([_paho_compat.PACKET_TOO_LARGE]))

    assert code.value == 149
    assert str(code) == "Packet too large"
    assert ReasonCode(packet, "Packet too large").value == 149
    assert ReasonCode(packet, identifier=149).is_failure


def test_the_fix_leaves_the_other_packets_names_alone() -> None:
    for packet in (PacketTypes.CONNACK, PacketTypes.DISCONNECT):
        assert str(ReasonCode(packet, identifier=149)) == "Packet too large"
    with pytest.raises(ValueError):
        ReasonCode(PacketTypes.SUBACK, identifier=149)


def test_installing_the_fix_twice_wraps_once() -> None:
    lookups = (ReasonCode.__getName__, ReasonCode.getId)
    _paho_compat.install()
    assert (ReasonCode.__getName__, ReasonCode.getId) == lookups


def test_a_refused_puback_completes_the_publish_instead_of_killing_the_loop() -> None:
    # paho's own PUBACK handler, fed the packet mosquitto sends for a payload
    # over message_size_limit. Unfixed, this raised out of the network thread.
    received = []
    client = mqtt.Client(
        callback_api_version=CallbackAPIVersion.VERSION2,
        client_id="payload-limit-test",
        protocol=mqtt.MQTTv5,
    )
    client.on_publish = lambda _c, _u, mid, reason_code, _p: received.append(
        (mid, reason_code.value)
    )
    info = client.publish("a/b", b"x", qos=1)
    client._in_packet = {
        "command": mqtt.PUBACK,
        "remaining_length": 3,
        "packet": struct.pack("!HB", info.mid, _paho_compat.PACKET_TOO_LARGE),
    }

    assert client._handle_pubackcomp("PUBACK") == mqtt.MQTT_ERR_SUCCESS
    assert received == [(info.mid, 149)]
    assert info.mid not in client._out_messages


# -- QoS 2: the refusal paho parses and then ignores -------------------------


def _qos2_client(reports: list) -> tuple[mqtt.Client, int]:
    client = mqtt.Client(
        callback_api_version=CallbackAPIVersion.VERSION2,
        client_id="payload-limit-test",
        protocol=mqtt.MQTTv5,
    )
    _paho_compat.report_refused_pubrecs(
        client, lambda mid, reason_code: reports.append((mid, reason_code))
    )
    return client, client.publish("a/b", b"x", qos=2).mid


def test_a_refused_pubrec_is_reported_and_paho_still_handles_it() -> None:
    # The PUBREC mosquitto sends for a QoS 2 payload over message_size_limit.
    # paho answers it with PUBREL and later reports the PUBCOMP's success to
    # on_publish, so without the hook the refusal is never seen.
    reports: list = []
    client, mid = _qos2_client(reports)
    published = []
    client.on_publish = lambda *args: published.append(args)
    client._in_packet = {
        "command": mqtt.PUBREC,
        "remaining_length": 3,
        "packet": struct.pack("!HB", mid, _paho_compat.PACKET_TOO_LARGE),
    }

    # No socket, so the PUBREL cannot leave; the hook passes paho's result on.
    assert client._handle_pubrec() == mqtt.MQTT_ERR_NO_CONN
    assert [(m, rc.value, str(rc)) for m, rc in reports] == [
        (mid, 149, "Packet too large")
    ]
    assert client._out_messages[mid].state == mqtt.mqtt_ms_wait_for_pubcomp
    assert published == []


def test_paho_dispatches_a_pubrec_through_the_hook() -> None:
    # The hook replaces the instance attribute, which only works while paho's
    # reader calls self._handle_pubrec(). Driven through _packet_handle, so a
    # paho that dispatches another way fails here instead of in production.
    reports: list = []
    client, mid = _qos2_client(reports)
    client._in_packet = {
        "command": mqtt.PUBREC,
        "remaining_length": 3,
        "packet": struct.pack("!HB", mid, _paho_compat.PACKET_TOO_LARGE),
    }

    assert client._packet_handle() == mqtt.MQTT_ERR_NO_CONN
    assert [(m, rc.value) for m, rc in reports] == [(mid, 149)]


@pytest.mark.parametrize(
    "remaining_length, body",
    [(3, b"\x00"), (2, b"")],
    ids=["v5-success", "no-reason-code"],
)
def test_an_accepting_pubrec_reports_nothing(remaining_length, body) -> None:
    reports: list = []
    client, mid = _qos2_client(reports)
    client._in_packet = {
        "command": mqtt.PUBREC,
        "remaining_length": remaining_length,
        "packet": struct.pack("!H", mid) + body,
    }

    assert client._handle_pubrec() == mqtt.MQTT_ERR_NO_CONN
    assert reports == []
    assert client._out_messages[mid].state == mqtt.mqtt_ms_wait_for_pubcomp


def test_reporting_refused_pubrecs_twice_wraps_once() -> None:
    client, _ = _qos2_client([])
    handler = client._handle_pubrec
    _paho_compat.report_refused_pubrecs(client, lambda *_: None)
    assert client._handle_pubrec is handler


def test_the_client_logs_a_refused_qos2_publish(caplog) -> None:
    client = _client(protocol=mqtt.MQTTv5)
    mid = client.client.publish("a/b", b"x", qos=2).mid
    client.client._in_packet = {
        "command": mqtt.PUBREC,
        "remaining_length": 3,
        "packet": struct.pack("!HB", mid, _paho_compat.PACKET_TOO_LARGE),
    }

    with caplog.at_level(logging.ERROR, logger="cyberwave.mqtt"):
        client.client._handle_pubrec()

    assert (
        f"MQTT broker refused a publish to a/b (mid={mid}): Packet too large (149)"
        in caplog.text
    )


# -- MQTT 5 properties -----------------------------------------------------


def test_v5_publish_properties_count_towards_the_packet() -> None:
    from paho.mqtt.properties import Properties

    properties = Properties(PacketTypes.PUBLISH)
    properties.ResponseTopic = "r" * 40
    properties.CorrelationData = b"c" * 32
    extra = len(properties.pack())
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = _accepting(mqtt_client_cls)
        client = _client(protocol=mqtt.MQTTv5)
        client._on_connect(mqtt_client, None, {}, 0, _connack(MaximumPacketSize=256))

        fits = 256 - len(TOPIC) - 16 - extra
        assert client.publish(TOPIC, "A" * fits, properties=properties) is True
        assert client.publish(TOPIC, "A" * (fits + 1), properties=properties) is False
        # Without the properties the same payload still fits.
        assert client.publish(TOPIC, "A" * (fits + 1)) is True
        assert mqtt_client.publish.call_count == 2


# -- the report throttle ---------------------------------------------------


def test_the_report_throttle_forgets_the_least_recently_reported_topic() -> None:
    now = [0.0]
    throttle = _payload_limit.ReportThrottle(10.0, max_keys=2, clock=lambda: now[0])

    assert throttle.admit("a") == 0
    assert throttle.admit("b") == 0
    assert throttle.admit("a") is None  # held back, and not made recent
    assert throttle.admit("c") == 0  # evicts "a", the least recently reported
    assert len(throttle) == 2
    # "a" was forgotten, so it is let through again; "b" is still held back.
    assert throttle.admit("a") == 0
    assert throttle.admit("c") is None
    now[0] = 10.0
    assert throttle.admit("c") == 1


def test_the_report_throttle_stays_bounded_under_many_topics() -> None:
    throttle = _payload_limit.ReportThrottle()
    for i in range(_payload_limit.REPORT_THROTTLE_MAX_KEYS * 4):
        throttle.admit(f"topic/{i}")
    assert len(throttle) == _payload_limit.REPORT_THROTTLE_MAX_KEYS


# -- raw paho paths --------------------------------------------------------


def test_reply_to_refuses_a_reply_over_the_limit(monkeypatch, caplog) -> None:
    from types import SimpleNamespace

    from paho.mqtt.properties import Properties

    from cyberwave.mqtt.request_response import reply_to

    monkeypatch.setenv(MAX_PAYLOAD_BYTES_ENV, "8")
    properties = Properties(PacketTypes.PUBLISH)
    properties.ResponseTopic = "a/reply"
    properties.CorrelationData = b"c"
    request = SimpleNamespace(topic="a/req", payload=b"{}", properties=properties)
    paho_client = MagicMock()

    with caplog.at_level(logging.ERROR, logger="cyberwave._payload_limit"):
        assert reply_to(paho_client, request, "A" * 9) is False
    paho_client.publish.assert_not_called()
    assert "Not replying on a/reply: payload is 9 bytes" in caplog.text

    assert reply_to(paho_client, request, "A" * 8) is True
    paho_client.publish.assert_called_once()


def test_reply_to_takes_the_limits_its_caller_knows(monkeypatch) -> None:
    """A raw paho client knows neither its owner's limit nor the broker's."""
    from types import SimpleNamespace

    from paho.mqtt.properties import Properties

    from cyberwave.mqtt.request_response import reply_to

    monkeypatch.setenv(MAX_PAYLOAD_BYTES_ENV, "8")
    properties = Properties(PacketTypes.PUBLISH)
    properties.ResponseTopic = "b/reply"
    properties.CorrelationData = b"c"
    request = SimpleNamespace(topic="b/req", payload=b"{}", properties=properties)
    paho_client = MagicMock()

    # The caller's limit wins over the environment's, both ways.
    assert reply_to(paho_client, request, "A" * 9, max_payload_bytes=9) is True
    assert reply_to(paho_client, request, "A" * 5, max_payload_bytes=4) is False
    assert reply_to(paho_client, request, "A" * 9, max_payload_bytes=0) is True
    # The packet, not just the payload, against the broker's CONNACK maximum.
    assert (
        reply_to(
            paho_client,
            request,
            "A",
            max_payload_bytes=0,
            broker_max_packet_size=10,
        )
        is False
    )
    assert paho_client.publish.call_count == 2


def _bridge(tmp_path, monkeypatch, limit: str):
    from cyberwave.zenoh_mqtt.bridge import ZenohMqttBridge
    from cyberwave.zenoh_mqtt.config import BridgeConfig

    monkeypatch.setenv(MAX_PAYLOAD_BYTES_ENV, limit)
    bridge = ZenohMqttBridge(
        config=BridgeConfig(enabled=True, queue_dir=str(tmp_path / "queue"))
    )
    bridge._mqtt_connected.set()
    bridge._mqtt_client = MagicMock()
    return bridge


def test_the_zenoh_bridge_drops_an_oversized_payload_before_buffering(
    tmp_path, monkeypatch, caplog
) -> None:
    bridge = _bridge(tmp_path, monkeypatch, "8")

    with caplog.at_level(logging.ERROR, logger="cyberwave.zenoh_mqtt.bridge"):
        bridge._publish_or_queue("cyberwave/twin/t/event", b"A" * 9)
        bridge._publish_or_queue("cyberwave/twin/t/event", b"A" * 8)
        bridge._drain_outbound_buffer()

    bridge._mqtt_client.publish.assert_called_once_with(
        "cyberwave/twin/t/event", b"A" * 8, qos=1
    )
    assert bridge.stats["oversized"] == 1
    assert bridge.stats["queued"] == 0
    assert "Not bridging to MQTT cyberwave/twin/t/event" in caplog.text


def test_the_zenoh_bridge_drops_an_oversized_queued_message_instead_of_retrying(
    tmp_path, monkeypatch
) -> None:
    import time

    from cyberwave.zenoh_mqtt.queue import QueuedMessage

    bridge = _bridge(tmp_path, monkeypatch, "8")
    # Queued earlier, by a bridge without the check.
    for payload in (b"A" * 9, b"A" * 8):
        bridge._queue.enqueue(
            QueuedMessage(
                mqtt_topic="cyberwave/twin/t/event",
                payload=payload,
                qos=1,
                enqueued_at=time.time(),
            )
        )

    worker = threading.Thread(target=bridge._drain_loop, daemon=True)
    worker.start()
    deadline = time.monotonic() + 5.0
    while bridge.stats["drained"] < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    bridge._stop_event.set()
    worker.join(timeout=5.0)

    bridge._mqtt_client.publish.assert_called_once_with(
        "cyberwave/twin/t/event", b"A" * 8, qos=1
    )
    assert bridge.stats["oversized"] == 1
    assert bridge._queue.is_empty
