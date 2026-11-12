"""MQTT 5 topic aliases: the topic is sent once, then referred to by number.

A telemetry topic is ~65 bytes and a pose payload ~150, so repeating the topic on
every message spends roughly a quarter of the packet restating a constant. These
tests pin what makes that safe: the first publish still carries the topic, later
ones must not, and a reconnect starts over. Aliasing is opt-in for now
(``topic_aliases=True`` or ``CYBERWAVE_MQTT_TOPIC_ALIASES=1``), so the helper
below opts in and the default is pinned separately.

Getting any of it wrong sends the broker an alias it never issued, a protocol
error it answers by closing the connection. The QoS and first-use tests below
are each one way that happened; the QoS one did it on every reconnect, forever.
"""

from __future__ import annotations

import socket
import threading
import time
from unittest.mock import MagicMock, patch

import paho.mqtt.client as mqtt
import pytest
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.properties import Properties

from cyberwave.mqtt import CyberwaveMQTTClient

TOPIC = "cyberwave/twin/8f1c2d3e-0000-4000-8000-000000000001/position"
OTHER = "cyberwave/twin/8f1c2d3e-0000-4000-8000-000000000001/rotation"


def _client(
    protocol: int = mqtt.MQTTv5, topic_aliases: bool | None = True
) -> CyberwaveMQTTClient:
    client = CyberwaveMQTTClient(
        mqtt_broker="localhost",
        mqtt_port=1883,
        mqtt_username="user",
        api_key="api_key_secret",
        auto_connect=False,
        protocol=protocol,
        topic_aliases=topic_aliases,
    )
    client.connected = True
    return client


def _connack(maximum: int) -> Properties:
    properties = Properties(PacketTypes.CONNACK)
    properties.TopicAliasMaximum = maximum
    return properties


def _published(mqtt_client: MagicMock) -> list[tuple[str, int | None]]:
    """(topic, alias) for every publish call."""
    calls = []
    for call in mqtt_client.publish.call_args_list:
        properties = call.kwargs.get("properties")
        alias = getattr(properties, "TopicAlias", None) if properties else None
        calls.append((call.args[0], alias))
    return calls


def _aliases_after_connack(**client_kwargs) -> list[tuple[str, int | None]]:
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = mqtt_client_cls.return_value
        mqtt_client.publish.return_value = MagicMock(rc=mqtt.MQTT_ERR_SUCCESS)
        client = _client(**client_kwargs)
        client._reset_topic_aliases(_connack(10))
        client.publish(TOPIC, {"x": 1})
        client.publish(TOPIC, {"x": 2})
        return _published(mqtt_client)


def test_aliasing_is_off_by_default_even_when_the_broker_offers_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Until NanoMQ stops offering aliases everywhere, see CyberwaveMQTTClient."""
    monkeypatch.delenv("CYBERWAVE_MQTT_TOPIC_ALIASES", raising=False)
    assert _aliases_after_connack(topic_aliases=None) == [
        (TOPIC, None),
        (TOPIC, None),
    ]


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_the_environment_variable_opts_in(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("CYBERWAVE_MQTT_TOPIC_ALIASES", value)
    assert _aliases_after_connack(topic_aliases=None) == [(TOPIC, 1), ("", 1)]


@pytest.mark.parametrize("value", ["", "0", "false", "off"])
def test_other_environment_values_leave_aliasing_off(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("CYBERWAVE_MQTT_TOPIC_ALIASES", value)
    assert _aliases_after_connack(topic_aliases=None) == [
        (TOPIC, None),
        (TOPIC, None),
    ]


def test_the_keyword_overrides_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CYBERWAVE_MQTT_TOPIC_ALIASES", "1")
    assert _aliases_after_connack(topic_aliases=False) == [
        (TOPIC, None),
        (TOPIC, None),
    ]
    monkeypatch.setenv("CYBERWAVE_MQTT_TOPIC_ALIASES", "0")
    assert _aliases_after_connack(topic_aliases=True) == [(TOPIC, 1), ("", 1)]


def test_the_first_publish_carries_the_topic_and_later_ones_do_not() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = mqtt_client_cls.return_value
        mqtt_client.publish.return_value = MagicMock(rc=mqtt.MQTT_ERR_SUCCESS)
        client = _client()
        client._reset_topic_aliases(_connack(10))

        client.publish(TOPIC, {"x": 1})
        client.publish(TOPIC, {"x": 2})
        client.publish(TOPIC, {"x": 3})

        # The mapping is established by sending both, exactly once. Sending an
        # empty topic before the broker has been told what alias 1 means would
        # be a protocol error, not an optimisation.
        assert _published(mqtt_client) == [(TOPIC, 1), ("", 1), ("", 1)]


def test_each_topic_gets_its_own_alias() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = mqtt_client_cls.return_value
        mqtt_client.publish.return_value = MagicMock(rc=mqtt.MQTT_ERR_SUCCESS)
        client = _client()
        client._reset_topic_aliases(_connack(10))

        client.publish(TOPIC, {"x": 1})
        client.publish(OTHER, {"y": 1})
        client.publish(TOPIC, {"x": 2})

        assert _published(mqtt_client) == [(TOPIC, 1), (OTHER, 2), ("", 1)]


def test_a_reconnect_forgets_every_alias() -> None:
    """An alias means nothing to a session that did not issue it."""
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = mqtt_client_cls.return_value
        mqtt_client.publish.return_value = MagicMock(rc=mqtt.MQTT_ERR_SUCCESS)
        client = _client()
        client._reset_topic_aliases(_connack(10))
        client.publish(TOPIC, {"x": 1})
        client.publish(TOPIC, {"x": 2})

        client._reset_topic_aliases(_connack(10))
        client.publish(TOPIC, {"x": 3})

        # Full topic again after the reconnect, or the publish would land on
        # whatever the new session happens to have numbered 1.
        assert _published(mqtt_client) == [(TOPIC, 1), ("", 1), (TOPIC, 1)]


def test_a_spent_budget_falls_back_to_the_full_topic() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = mqtt_client_cls.return_value
        mqtt_client.publish.return_value = MagicMock(rc=mqtt.MQTT_ERR_SUCCESS)
        client = _client()
        client._reset_topic_aliases(_connack(1))

        client.publish(TOPIC, {"x": 1})
        client.publish(OTHER, {"y": 1})
        client.publish(OTHER, {"y": 2})

        assert _published(mqtt_client) == [(TOPIC, 1), (OTHER, None), (OTHER, None)]


def test_a_broker_that_allows_none_disables_aliasing() -> None:
    """``TopicAliasMaximum`` absent from CONNACK means zero, not unlimited."""
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = mqtt_client_cls.return_value
        mqtt_client.publish.return_value = MagicMock(rc=mqtt.MQTT_ERR_SUCCESS)
        client = _client()
        client._reset_topic_aliases(_connack(0))

        client.publish(TOPIC, {"x": 1})
        client.publish(TOPIC, {"x": 2})

        assert _published(mqtt_client) == [(TOPIC, None), (TOPIC, None)]


def test_mqtt_v311_never_aliases() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = mqtt_client_cls.return_value
        mqtt_client.publish.return_value = MagicMock(rc=mqtt.MQTT_ERR_SUCCESS)
        client = _client(protocol=mqtt.MQTTv311)
        client._reset_topic_aliases(_connack(10))

        client.publish(TOPIC, {"x": 1})
        client.publish(TOPIC, {"x": 2})

        assert _published(mqtt_client) == [(TOPIC, None), (TOPIC, None)]


# -- when an alias must not be used -------------------------------------------


def test_caller_properties_go_out_as_given_without_an_alias() -> None:
    """Request/response publishes carry their own v5 properties untouched."""
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = mqtt_client_cls.return_value
        mqtt_client.publish.return_value = MagicMock(rc=mqtt.MQTT_ERR_SUCCESS)
        client = _client()
        client._reset_topic_aliases(_connack(10))
        client.publish(TOPIC, {"x": 1})

        own = Properties(PacketTypes.PUBLISH)
        own.ResponseTopic = "cyberwave/reply/abc"
        own.CorrelationData = b"\x01\x02"
        client.publish(TOPIC, {"x": 2}, properties=own)

        call = mqtt_client.publish.call_args
        assert call.args[0] == TOPIC
        assert call.kwargs["properties"] is own
        assert getattr(own, "TopicAlias", None) is None


def test_qos_1_and_2_always_carry_the_full_topic() -> None:
    """paho resends these after a reconnect exactly as first published.

    With an alias, that resend is an empty topic plus a number from the old
    session: the new broker session closes the connection, paho reconnects and
    resends it again, and the client never gets past that message.
    """
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = mqtt_client_cls.return_value
        mqtt_client.publish.return_value = MagicMock(rc=mqtt.MQTT_ERR_SUCCESS)
        client = _client()
        client._reset_topic_aliases(_connack(10))

        client.publish(TOPIC, {"x": 1}, qos=1)
        client.publish(TOPIC, {"x": 2}, qos=2)
        client.publish(TOPIC, {"x": 3}, qos=0)
        client.publish(TOPIC, {"x": 4}, qos=0)
        client.publish(TOPIC, {"x": 5}, qos=1)

        # QoS 0 still aliases, and QoS 1/2 took no alias of their own: the
        # first QoS 0 publish is the one that establishes alias 1.
        assert _published(mqtt_client) == [
            (TOPIC, None),
            (TOPIC, None),
            (TOPIC, 1),
            ("", 1),
            (TOPIC, None),
        ]
        assert all(
            "properties" not in call.kwargs
            for call in mqtt_client.publish.call_args_list
            if call.kwargs["qos"]
        )


def test_an_in_flight_qos_1_message_keeps_its_topic_in_paho() -> None:
    """Regression guard, against real paho, for what it resends after a reconnect.

    paho keeps a QoS 1 publish in ``_out_messages`` until it is acked and
    resends ``m.topic`` with ``m.properties`` from there. Aliasing QoS 1 would
    put an empty topic and an alias from the old session in that store; this
    pins that the stored message still names its topic and carries no alias.
    """
    client = _client()
    client._reset_topic_aliases(_connack(10))

    client.publish(TOPIC, {"x": 2}, qos=1)

    [kept] = client.client._out_messages.values()
    assert kept.topic == TOPIC
    assert getattr(kept.properties, "TopicAlias", None) is None


def test_a_refused_first_publish_does_not_establish_the_alias() -> None:
    """The broker never saw it, so it knows no alias for the topic yet."""
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = mqtt_client_cls.return_value
        mqtt_client.publish.side_effect = [
            MagicMock(rc=mqtt.MQTT_ERR_NO_CONN),
            MagicMock(rc=mqtt.MQTT_ERR_SUCCESS),
            MagicMock(rc=mqtt.MQTT_ERR_SUCCESS),
        ]
        client = _client()
        client._reset_topic_aliases(_connack(10))

        assert client.publish(TOPIC, {"x": 1}) is False
        assert client.publish(TOPIC, {"x": 2}) is True
        assert client.publish(TOPIC, {"x": 3}) is True

        assert _published(mqtt_client) == [(TOPIC, 1), (TOPIC, 1), ("", 1)]


def test_a_refused_first_publish_against_real_paho_leaves_no_alias() -> None:
    client = _client()
    client._reset_topic_aliases(_connack(10))

    # No socket: paho refuses a QoS 0 publish outright.
    assert client.publish(TOPIC, {"x": 1}) is False
    assert client._topic_aliases == {}


def test_a_disconnect_forgets_every_alias_before_the_next_connack() -> None:
    """paho reconnects before CONNACK; a publish in between is a first use."""
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = mqtt_client_cls.return_value
        mqtt_client.publish.return_value = MagicMock(rc=mqtt.MQTT_ERR_SUCCESS)
        client = _client()
        client._reset_topic_aliases(_connack(10))
        client.publish(TOPIC, {"x": 1})

        client._on_disconnect(mqtt_client, None, 0)
        # A publish that passed the ``connected`` check just before the drop.
        client.connected = True
        client.publish(TOPIC, {"x": 2})
        # And aliasing resumes only once the new session says it may.
        client._reset_topic_aliases(_connack(10))
        client.publish(TOPIC, {"x": 3})
        client.publish(TOPIC, {"x": 4})

        assert _published(mqtt_client) == [
            (TOPIC, 1),
            (TOPIC, None),
            (TOPIC, 1),
            ("", 1),
        ]


def test_concurrent_first_uses_send_the_topic_before_the_alias() -> None:
    """The mapping a first use makes is not visible until it is enqueued.

    The first publish is held inside paho long enough for a second thread to
    publish the same topic. Had the second thread seen the new mapping, its
    empty-topic publish would be enqueued -- and on the wire -- first.
    """
    enqueued: list[tuple[str, int | None]] = []
    first_inside = threading.Event()

    def enqueue(topic, payload, qos=0, properties=None):
        if not enqueued and not first_inside.is_set():
            first_inside.set()
            time.sleep(0.1)
        enqueued.append((topic, getattr(properties, "TopicAlias", None)))
        return MagicMock(rc=mqtt.MQTT_ERR_SUCCESS)

    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client_cls.return_value.publish.side_effect = enqueue
        client = _client()
        client._reset_topic_aliases(_connack(10))

        first = threading.Thread(target=client.publish, args=(TOPIC, {"x": 1}))
        first.start()
        assert first_inside.wait(5)
        second = threading.Thread(target=client.publish, args=(TOPIC, {"x": 2}))
        second.start()
        first.join(5)
        second.join(5)

    assert enqueued == [(TOPIC, 1), ("", 1)]


def test_the_brokers_connack_reaches_the_alias_budget_through_paho() -> None:
    """Every other test hands `_reset_topic_aliases` its properties directly.

    This one feeds paho a CONNACK packet, so paho's own callback dispatch decides
    which argument of `_on_connect` carries the properties.
    """
    client = _client()
    client.connected = False
    packet = b"\x00\x00" + _connack(10).pack()  # session flags, reason code 0
    client.client._in_packet = {
        "command": mqtt.CONNACK,
        "remaining_length": len(packet),
        "packet": packet,
    }

    client.client._handle_connack()

    assert client.connected
    assert client._topic_alias_maximum == 10


def test_a_disconnect_from_inside_publish_neither_deadlocks_nor_keeps_the_alias() -> None:
    """With the loop stopped, paho writes on the publishing thread.

    A failed write then calls ``_on_disconnect`` -- and so the alias reset --
    from inside ``client.publish``, while ``_publish_aliased`` holds the lock.
    """
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = mqtt_client_cls.return_value
        client = _client()
        client._reset_topic_aliases(_connack(10))

        def write_fails(topic, payload, qos=0, properties=None):
            client._on_disconnect(mqtt_client, None, 0)
            return MagicMock(rc=mqtt.MQTT_ERR_SUCCESS)

        mqtt_client.publish.side_effect = write_fails
        done = threading.Event()
        threading.Thread(
            target=lambda: (client.publish(TOPIC, {"x": 1}), done.set()),
            daemon=True,
        ).start()

        assert done.wait(5), "publish deadlocked on the alias lock"
        # The reset ran after the alias was chosen; recording it afterwards
        # would hand the next session an alias it never saw.
        assert client._topic_aliases == {}
        assert client._topic_alias_maximum == 0


def test_paho_reconnect_forgets_every_alias() -> None:
    """Against real paho: reconnect() does not call on_disconnect.

    A caller's watchdog (the ROS 2 bridge) and the retry loop in connect()
    both call it directly; on_pre_connect is what clears the old session's
    aliases before the new socket opens.
    """
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    try:
        client = _client()
        client.client.connect_async("127.0.0.1", listener.getsockname()[1])
        client._reset_topic_aliases(_connack(10))
        client._topic_aliases[TOPIC] = 1

        client.client.reconnect()

        assert client._topic_aliases == {}
        assert client._topic_alias_maximum == 0
    finally:
        client.client._sock_close()
        listener.close()


def test_a_connack_during_the_retry_backoff_is_kept_not_reconnected() -> None:
    """The first attempt's CONNACK can land after its wait, during the backoff.

    reconnect() on the next attempt would tear that healthy session down and
    start over.
    """
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = mqtt_client_cls.return_value
        client = _client()
        client.connected = False
        client._connect_timeout = 0.01
        client._connect_max_attempts = 3

        def sleep(seconds: float) -> None:
            if seconds >= 1.0:  # the backoff, not the CONNACK poll
                client.connected = True

        with patch("cyberwave.mqtt.time.sleep", side_effect=sleep):
            client.connect()

        mqtt_client.connect.assert_called_once()
        mqtt_client.reconnect.assert_not_called()
        assert client.connected
