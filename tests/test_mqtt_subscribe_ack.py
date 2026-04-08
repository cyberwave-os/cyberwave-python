"""Tests for broker-level MQTT subscribe acknowledgements."""

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
