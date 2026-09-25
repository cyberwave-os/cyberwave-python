"""MQTT v5 no_local subscription for joint/update echo filtering."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import paho.mqtt.client as mqtt

from cyberwave.config import CyberwaveConfig, _parse_mqtt_protocol_env
from cyberwave.mqtt import CyberwaveMQTTClient


def test_parse_mqtt_protocol_env() -> None:
    assert _parse_mqtt_protocol_env("5") == mqtt.MQTTv5
    assert _parse_mqtt_protocol_env("mqttv5") == mqtt.MQTTv5
    assert _parse_mqtt_protocol_env("311") == mqtt.MQTTv311


def test_config_loads_mqtt_protocol_from_env() -> None:
    with patch.dict("os.environ", {"CYBERWAVE_MQTT_PROTOCOL": "5"}):
        config = CyberwaveConfig()
    assert config.mqtt_protocol == mqtt.MQTTv5


def test_subscribe_uses_no_local_options_on_mqtt_v5() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = mqtt_client_cls.return_value
        mqtt_client.subscribe.return_value = (0, 7)

        client = CyberwaveMQTTClient(
            mqtt_broker="localhost",
            mqtt_port=1883,
            mqtt_username="user",
            api_key="api_key_secret",
            auto_connect=False,
            protocol=mqtt.MQTTv5,
        )
        client.connected = True

        client.subscribe(
            "cyberwave/joint/twin-uuid/update",
            handler=MagicMock(),
            no_local=True,
        )

    assert client.is_mqtt_v5
    _args, kwargs = mqtt_client.subscribe.call_args
    assert kwargs["options"].noLocal is True
    assert "cyberwave/joint/twin-uuid/update" in client._subscribe_options


def test_subscribe_no_local_is_noop_on_mqtt_v311() -> None:
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = mqtt_client_cls.return_value
        mqtt_client.subscribe.return_value = (0, 8)

        client = CyberwaveMQTTClient(
            mqtt_broker="localhost",
            mqtt_port=1883,
            mqtt_username="user",
            api_key="api_key_secret",
            auto_connect=False,
            protocol=mqtt.MQTTv311,
        )
        client.connected = True

        client.subscribe(
            "cyberwave/joint/twin-uuid/update",
            handler=MagicMock(),
            no_local=True,
        )

    mqtt_client.subscribe.assert_called_once_with(
        "cyberwave/joint/twin-uuid/update",
        qos=0,
    )
    assert "cyberwave/joint/twin-uuid/update" not in client._subscribe_options
