"""Tests for broker-level MQTT subscribe acknowledgements."""

import threading
from unittest.mock import patch

from cyberwave.mqtt import CyberwaveMQTTClient


class _FakeReasonCode:
    def __init__(self, value: int, label: str) -> None:
        self.value = value
        self._label = label

    def __str__(self) -> str:
        return self._label


def test_subscribe_logs_broker_rejection_reason(caplog):
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = mqtt_client_cls.return_value
        mqtt_client.subscribe.return_value = (0, 42)

        client = CyberwaveMQTTClient(
            mqtt_broker="localhost",
            mqtt_port=1883,
            mqtt_username="user",
            api_key="api_key_secret",
            auto_connect=False,
        )
        client.connected = True

        client.subscribe("cyberwave/joint/test-twin/+")
        client._on_subscribe(
            mqtt_client,
            None,
            42,
            [_FakeReasonCode(135, "Not authorized")],
        )

    assert "SUBACK rejected subscription for cyberwave/joint/test-twin/+" in caplog.text
    assert "Not authorized" in caplog.text
    assert 42 not in client._pending_subscriptions


def _client_with_mids(*mids):
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = mqtt_client_cls.return_value
        mqtt_client.subscribe.side_effect = [(0, mid) for mid in mids]
        client = CyberwaveMQTTClient(
            mqtt_broker="localhost",
            mqtt_port=1883,
            mqtt_username="user",
            api_key="api_key_secret",
            auto_connect=False,
        )
    client.connected = True
    return client, mqtt_client


def test_a_subscription_counts_only_once_the_broker_acks_it():
    client, mqtt_client = _client_with_mids(7)
    topic = "cyberwave/twin/t/webrtc-answer"

    client.subscribe(topic, lambda _data: None)
    assert not client.is_subscribed(topic)

    client._on_subscribe(mqtt_client, None, 7, [_FakeReasonCode(0, "Granted QoS 0")])
    assert client.is_subscribed(topic)


def test_a_rejected_subscription_never_counts():
    client, mqtt_client = _client_with_mids(8)
    topic = "cyberwave/twin/t/webrtc-answer"

    client.subscribe(topic, lambda _data: None)
    client._on_subscribe(mqtt_client, None, 8, [_FakeReasonCode(135, "Not authorized")])

    assert not client.is_subscribed(topic)


def test_a_reconnect_or_unsubscribe_drops_the_ack():
    client, mqtt_client = _client_with_mids(1, 2, 3)
    topic = "cyberwave/twin/t/webrtc-answer"
    client.subscribe(topic, lambda _data: None)
    client._on_subscribe(mqtt_client, None, 1, [_FakeReasonCode(0, "Granted QoS 0")])

    # A new broker session: resubscribed, but not acked yet.
    client._on_connect(mqtt_client, None, None, 0)
    assert not client.is_subscribed(topic)
    client._on_subscribe(mqtt_client, None, 2, [_FakeReasonCode(0, "Granted QoS 0")])
    assert client.is_subscribed(topic)

    client.unsubscribe(topic)
    assert not client.is_subscribed(topic)


def test_another_handler_on_an_acked_topic_keeps_the_ack():
    client, mqtt_client = _client_with_mids(4)
    topic = "cyberwave/twin/t/webrtc-answer"
    client.subscribe(topic, lambda _data: None, subscriber_key="mic")
    client._on_subscribe(mqtt_client, None, 4, [_FakeReasonCode(0, "Granted QoS 0")])

    # Shares the broker subscription: no SUBSCRIBE, so nothing to wait for.
    client.subscribe(topic, lambda _data: None, subscriber_key="speaker")

    assert client.is_subscribed(topic)
    assert mqtt_client.subscribe.call_count == 1


def test_a_suback_racing_the_subscribe_call_still_counts():
    """paho can deliver the SUBACK on its network thread before ``subscribe``
    returns the mid; the topic must not be left pending for good."""
    import threading

    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = mqtt_client_cls.return_value
        client = CyberwaveMQTTClient(
            mqtt_broker="localhost",
            mqtt_port=1883,
            mqtt_username="user",
            api_key="api_key_secret",
            auto_connect=False,
        )
    client.connected = True
    topic = "cyberwave/twin/t/webrtc-answer"
    network_thread: list[threading.Thread] = []

    def subscribe_with_instant_suback(*_args, **_kwargs):
        ack = threading.Thread(
            target=client._on_subscribe,
            args=(mqtt_client, None, 9, [_FakeReasonCode(0, "Granted QoS 0")]),
        )
        network_thread.append(ack)
        ack.start()
        ack.join(timeout=0.2)  # give the SUBACK every chance to win the race
        return (0, 9)

    mqtt_client.subscribe.side_effect = subscribe_with_instant_suback

    client.subscribe(topic, lambda _data: None)
    network_thread[0].join(timeout=2.0)

    assert client.is_subscribed(topic)
    assert 9 not in client._pending_subscriptions


def test_a_subscription_made_while_offline_counts_only_after_its_suback():
    client, mqtt_client = _client_with_mids(5)
    client.connected = False
    topic = "cyberwave/twin/t/webrtc-answer"

    client.subscribe(topic, lambda _data: None)
    assert not client.is_subscribed(topic)

    client._on_connect(mqtt_client, None, None, 0)
    assert not client.is_subscribed(topic)
    client._on_subscribe(mqtt_client, None, 5, [_FakeReasonCode(0, "Granted QoS 0")])
    assert client.is_subscribed(topic)


def test_a_reconnect_never_reports_a_topic_it_has_not_resubscribed_yet():
    """``connected`` is True before the resubscribes go out. Another thread
    polling ``is_subscribed`` then (a WebRTC offer waiting out the outage) must
    not see the answer topic as acked before its new SUBSCRIBE is even sent."""
    client, _ = _client_with_mids()
    first = "cyberwave/twin/t/command"
    answer = "cyberwave/twin/t/webrtc-answer"
    client._handlers[first] = {None: lambda _data: None}
    client._handlers[answer] = {None: lambda _data: None}
    client.connected = False

    seen: list[bool] = []
    poller = threading.Thread(target=lambda: seen.append(client.is_subscribed(answer)))
    mids = iter(range(100, 200))

    def subscribe(topic, **_kwargs):
        if topic == first:
            # The answer topic's SUBSCRIBE has not been sent yet.
            poller.start()
            poller.join(0.2)
        return (0, next(mids))

    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        paho = mqtt_client_cls.return_value
        paho.subscribe.side_effect = subscribe
        client._on_connect(paho, None, None, 0)
    poller.join(1.0)

    assert seen == [False]
    assert not client.is_subscribed(answer)


def test_a_reconnect_is_never_reported_before_its_resubscribes_are_sent():
    """The instant ``connected`` turns True on a reconnect, a topic acked on the
    previous connection must not read as subscribed: its new SUBSCRIBE has not
    gone out yet, so an offer published then would lose its answer."""
    client, _ = _client_with_mids()
    answer = "cyberwave/twin/t/webrtc-answer"
    client._handlers[answer] = {None: lambda _data: None}
    client.connected = False

    seen: list[bool] = []
    pollers: list[threading.Thread] = []

    def set_connected(self, value):
        self.__dict__["_connected_for_test"] = value
        if value:
            poller = threading.Thread(
                target=lambda: seen.append(client.is_subscribed(answer))
            )
            pollers.append(poller)
            poller.start()
            poller.join(0.2)

    client.__class__ = type(
        "_ConnectedProbe",
        (CyberwaveMQTTClient,),
        {
            "connected": property(
                lambda self: self.__dict__.get("_connected_for_test", False),
                set_connected,
            )
        },
    )

    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        paho = mqtt_client_cls.return_value
        paho.subscribe.return_value = (0, 300)
        client._on_connect(paho, None, None, 0)
    for poller in pollers:
        poller.join(1.0)

    assert seen == [False]
    assert not client.is_subscribed(answer)
