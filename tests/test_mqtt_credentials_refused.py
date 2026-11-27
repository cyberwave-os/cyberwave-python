"""A broker that refuses the credentials is not retried every few seconds."""

from __future__ import annotations

import logging
from unittest.mock import call, patch

import paho.mqtt.client as paho_mqtt
import pytest
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.reasoncodes import ReasonCode

from cyberwave.exceptions import MQTTCredentialsRejectedError
from cyberwave.mqtt import CyberwaveMQTTClient

NOT_AUTHORIZED = ReasonCode(PacketTypes.CONNACK, identifier=0x87)


def _client(**kwargs) -> CyberwaveMQTTClient:
    return CyberwaveMQTTClient(**{"api_key": "test-api-key", **kwargs})


def _delays(paho: paho_mqtt.Client) -> tuple[int, int]:
    return paho._reconnect_min_delay, paho._reconnect_max_delay


@pytest.mark.parametrize(
    "rc",
    [
        NOT_AUTHORIZED,
        ReasonCode(PacketTypes.CONNACK, identifier=0x86),
        # MQTT 3.1.1's return codes, as paho hands them to on_connect.
        paho_mqtt.convert_connack_rc_to_reason_code(4),
        paho_mqtt.convert_connack_rc_to_reason_code(5),
    ],
    ids=str,
)
def test_a_credentials_refusal_slows_reconnects(rc) -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as paho_client_cls:
        client = _client()
        paho = paho_client_cls.return_value

        client._on_connect(paho, None, {}, rc)

    paho.reconnect_delay_set.assert_called_once_with(min_delay=60, max_delay=60)
    assert not client.connected


def test_refusals_in_a_row_back_off_to_fifteen_minutes() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as paho_client_cls:
        client = _client()
        paho = paho_client_cls.return_value

        for _ in range(4):
            client._on_connect(paho, None, {}, NOT_AUTHORIZED)

    assert paho.reconnect_delay_set.call_args_list == [
        call(min_delay=delay, max_delay=delay) for delay in (60, 300, 900, 900)
    ]


@pytest.mark.parametrize(
    ("kwargs", "credential"),
    [
        ({}, "API key (CYBERWAVE_API_KEY)"),
        ({"mqtt_password": "pw"}, "MQTT password (CYBERWAVE_MQTT_PASSWORD)"),
    ],
)
def test_a_refusal_is_logged_once_and_names_the_credential(
    caplog: pytest.LogCaptureFixture, kwargs, credential
) -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as paho_client_cls:
        client = _client(**kwargs)
        paho = paho_client_cls.return_value

        with caplog.at_level(logging.ERROR, logger="cyberwave.mqtt"):
            for _ in range(3):
                client._on_connect(paho, None, {}, NOT_AUTHORIZED)

    assert [r.getMessage() for r in caplog.records] == [
        f"MQTT broker {client.mqtt_broker}:{client.mqtt_port} rejected the "
        f"credentials (Not authorized): check the {credential}"
    ]


def test_an_accepted_connection_restores_the_callers_backoff() -> None:
    client = _client()
    paho = client.client  # a real paho client, never connected
    # Set on paho directly, as edge-core does for its shared client.
    paho.reconnect_delay_set(min_delay=1, max_delay=30)

    client._on_connect(paho, None, {}, NOT_AUTHORIZED)
    client._on_connect(paho, None, {}, NOT_AUTHORIZED)
    assert _delays(paho) == (300, 300)

    client._on_connect(paho, None, {}, 0)
    assert _delays(paho) == (1, 30)

    client._on_connect(paho, None, {}, NOT_AUTHORIZED)
    assert _delays(paho) == (60, 60)


def test_a_background_connect_after_a_refused_connect_keeps_its_backoff() -> None:
    # The fallback Cyberwave.run_workers takes when connect() raises.
    client = _client()
    paho = client.client  # a real paho client; its network calls are stubbed

    def refuse(*_args, **_kwargs) -> None:
        client._on_connect(paho, None, {}, NOT_AUTHORIZED)

    with (
        patch.object(paho, "connect", side_effect=refuse),
        patch.object(paho, "connect_async"),
        patch.object(paho, "loop_start"),
        patch.object(paho, "loop_stop"),
    ):
        with pytest.raises(MQTTCredentialsRejectedError):
            client.connect()
        client.connect_in_background()
        assert _delays(paho) == (1, 30)

        refuse()
        assert _delays(paho) == (60, 60)

        client._on_connect(paho, None, {}, 0)
        assert _delays(paho) == (1, 30)


def test_disconnecting_ends_the_refusals() -> None:
    client = _client()
    paho = client.client
    with patch.object(paho, "connect_async"), patch.object(paho, "loop_start"):
        client.connect_in_background()
    client._on_connect(paho, None, {}, NOT_AUTHORIZED)
    client._on_connect(paho, None, {}, NOT_AUTHORIZED)

    with patch.object(paho, "loop_stop"), patch.object(paho, "disconnect"):
        client.disconnect()

    assert _delays(paho) == (1, 30)
    client._on_connect(paho, None, {}, NOT_AUTHORIZED)
    assert _delays(paho) == (60, 60)


@pytest.mark.parametrize("identifier", [0x88, 0x89])
def test_other_refusals_keep_the_backoff(identifier) -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as paho_client_cls:
        client = _client()
        paho = paho_client_cls.return_value

        client._on_connect(
            paho, None, {}, ReasonCode(PacketTypes.CONNACK, identifier=identifier)
        )

    paho.reconnect_delay_set.assert_not_called()


def test_connect_raises_at_the_first_credentials_refusal() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as paho_client_cls:
        client = _client()
        paho = paho_client_cls.return_value
        paho.connect.side_effect = lambda *a, **k: client._on_connect(
            paho, None, {}, NOT_AUTHORIZED
        )

        with pytest.raises(MQTTCredentialsRejectedError, match="Not authorized"):
            client.connect()

    paho.reconnect.assert_not_called()
    paho.loop_stop.assert_called_once_with()


def test_connect_raises_the_refusal_of_a_background_connect() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as paho_client_cls:
        client = _client()
        paho = paho_client_cls.return_value
        client.connect_in_background()
        client._on_connect(paho, None, {}, NOT_AUTHORIZED)

        # Without waiting out the timeout: paho will not try again for a minute.
        with pytest.raises(MQTTCredentialsRejectedError):
            client.connect()

    paho.connect.assert_not_called()
