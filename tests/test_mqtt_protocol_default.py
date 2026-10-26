"""MQTT v5 is the default protocol; 3.1.1 stays available as an opt-out."""

from __future__ import annotations

import logging
from unittest.mock import patch

import paho.mqtt.client as paho
import pytest

from cyberwave.config import CyberwaveConfig
from cyberwave.mqtt import CyberwaveMQTTClient


def _client(**kwargs) -> CyberwaveMQTTClient:
    return CyberwaveMQTTClient(api_key="test-api-key", **kwargs)


def test_a_client_connects_with_mqtt_v5_by_default() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as paho_client_cls:
        client = _client()

    assert client.is_mqtt_v5
    assert paho_client_cls.call_args.kwargs["protocol"] == paho.MQTTv5


def test_a_client_can_still_ask_for_mqtt_311() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as paho_client_cls:
        client = _client(protocol=paho.MQTTv311)

    assert not client.is_mqtt_v5
    assert paho_client_cls.call_args.kwargs["protocol"] == paho.MQTTv311


def test_config_leaves_the_protocol_to_the_client_default(monkeypatch) -> None:
    monkeypatch.delenv("CYBERWAVE_MQTT_PROTOCOL", raising=False)

    assert CyberwaveConfig().mqtt_protocol is None


def test_the_environment_can_opt_back_to_mqtt_311(monkeypatch) -> None:
    monkeypatch.setenv("CYBERWAVE_MQTT_PROTOCOL", "311")

    assert CyberwaveConfig().mqtt_protocol == paho.MQTTv311


@pytest.mark.parametrize(
    ("protocol", "expects_hint"), [(paho.MQTTv5, True), (paho.MQTTv311, False)]
)
def test_a_failed_v5_connect_says_how_to_fall_back(
    protocol, expects_hint, caplog
) -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as paho_client_cls:
        paho_client_cls.return_value.connect.side_effect = OSError("refused")
        client = _client(protocol=protocol)
        client._connect_max_attempts = 1

        with caplog.at_level(logging.ERROR, logger="cyberwave.mqtt"):
            with pytest.raises(OSError):
                client.connect()

    hinted = "CYBERWAVE_MQTT_PROTOCOL=311" in caplog.text
    assert hinted is expects_hint
