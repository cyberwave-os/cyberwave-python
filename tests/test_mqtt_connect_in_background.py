"""A client can keep connecting from paho's network thread after connect() fails."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from cyberwave.mqtt import CyberwaveMQTTClient


def _client() -> CyberwaveMQTTClient:
    return CyberwaveMQTTClient(api_key="test-api-key")


def test_connect_in_background_hands_connecting_to_the_network_thread() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as paho_client_cls:
        client = _client()
        paho = paho_client_cls.return_value

        client.connect_in_background()
        client.connect_in_background()

    paho.connect_async.assert_called_once_with(
        client.mqtt_broker, client.mqtt_port, keepalive=60
    )
    paho.loop_start.assert_called_once_with()
    assert client.connecting_in_background


def test_disconnect_stops_a_background_connect() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as paho_client_cls:
        client = _client()
        paho = paho_client_cls.return_value
        client.connect_in_background()

        client.disconnect()

    paho.loop_stop.assert_called_once_with()
    assert not client.connecting_in_background


def test_connect_leaves_a_background_connect_to_the_network_thread() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as paho_client_cls:
        client = _client()
        client._connect_timeout = 0
        paho = paho_client_cls.return_value
        client.connect_in_background()

        # Still "connected or raised": it waits for the network thread and
        # raises while the broker has not answered.
        with pytest.raises(Exception, match="retrying in the background"):
            client.connect()

    paho.connect.assert_not_called()
    paho.loop_stop.assert_not_called()
    assert client.connecting_in_background


def test_connect_returns_once_the_background_connect_succeeds() -> None:
    with patch("cyberwave.mqtt.mqtt.Client"):
        client = _client()
        client.connect_in_background()
        client.connected = True

        client.connect()


def test_background_connect_caps_the_reconnect_backoff() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as paho_client_cls:
        client = _client()
        paho = paho_client_cls.return_value

        client.connect_in_background()

    paho.reconnect_delay_set.assert_called_once_with(min_delay=1, max_delay=30)
    assert paho.on_connect_fail == client._on_background_connect_fail


def test_a_topic_subscribed_offline_keeps_its_qos_on_connect() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as paho_client_cls:
        client = _client()
        paho = paho_client_cls.return_value
        paho.subscribe.return_value = (0, 1)

        client.subscribe("t/run", lambda _data: None, qos=1)
        client._on_connect(paho, None, {}, 0)

    paho.subscribe.assert_called_once_with("t/run", qos=1)


def test_subscribe_rejects_an_invalid_qos_before_storing_it() -> None:
    client = _client()

    with pytest.raises(ValueError):
        client.subscribe("t/bad", lambda *_: None, qos=3)

    assert "t/bad" not in client._handlers
    assert "t/bad" not in client._subscribe_qos


def test_background_connect_that_fails_to_start_can_be_retried() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as paho_client_cls:
        client = _client()
        paho_client_cls.return_value.loop_start.side_effect = OSError(
            "no file descriptors"
        )

        with pytest.raises(OSError):
            client.connect_in_background()

    assert client.connecting_in_background is False


def test_background_connect_announces_twins_registered_while_offline() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as paho_client_cls:
        client = CyberwaveMQTTClient(api_key="test-api-key", twin_uuids=["twin-1"])
        paho = paho_client_cls.return_value
        paho.subscribe.return_value = (0, 1)
        client.connect_in_background()

        with patch.object(client, "_publish_connect_message") as connect_msg, patch.object(
            client, "_publish_telemetry_start_message"
        ) as start_msg:
            client._on_connect(paho, None, {}, 0)
            client.connect()

    connect_msg.assert_called_once_with("twin-1")
    start_msg.assert_called_once_with("twin-1", None)


def test_connect_after_a_drop_leaves_reconnecting_to_the_network_thread() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as paho_client_cls:
        client = _client()
        client._connect_timeout = 0
        paho = paho_client_cls.return_value
        paho.connect.side_effect = lambda *a, **k: setattr(client, "connected", True)
        client.connect()
        client.connected = False  # the connection dropped; paho is reconnecting

        with pytest.raises(Exception, match="retrying in the background"):
            client.connect()

    paho.connect.assert_called_once()
    paho.reconnect.assert_not_called()
    paho.loop_start.assert_called_once_with()


def test_disconnect_after_a_drop_stops_the_network_thread() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as paho_client_cls:
        client = _client()
        paho = paho_client_cls.return_value
        paho.connect.side_effect = lambda *a, **k: setattr(client, "connected", True)
        client.connect()
        client.connected = False

        client.disconnect()

    paho.loop_stop.assert_called_once_with()


@pytest.mark.parametrize("qos", [1.0, True, "1"])
def test_subscribe_rejects_a_qos_that_is_not_an_int(qos) -> None:
    with patch("cyberwave.mqtt.mqtt.Client"):
        client = _client()
        with pytest.raises(ValueError, match="Invalid MQTT QoS"):
            client.subscribe("some/topic", lambda _data: None, qos=qos)


def test_connect_in_background_leaves_a_dropped_connect_to_its_network_thread() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as paho_client_cls:
        client = _client()
        paho = paho_client_cls.return_value
        paho.connect.side_effect = lambda *a, **k: setattr(client, "connected", True)
        client.connect()
        client.connected = False

        client.connect_in_background()

    paho.connect_async.assert_not_called()
    paho.loop_start.assert_called_once_with()


def test_wrapper_reports_a_dropped_connect_as_reconnecting_in_background() -> None:
    """The worker hook skips connect() then, instead of waiting out its timeout."""
    from cyberwave.mqtt_client import CyberwaveMQTTClient as WrapperClient

    with patch("cyberwave.mqtt.mqtt.Client") as paho_client_cls:
        client = _client()
        paho_client_cls.return_value.connect.side_effect = (
            lambda *a, **k: setattr(client, "connected", True)
        )
        wrapper = object.__new__(WrapperClient)
        wrapper._client = client
        assert not wrapper.connecting_in_background
        client.connect()
        client.connected = False

        assert wrapper.connecting_in_background
