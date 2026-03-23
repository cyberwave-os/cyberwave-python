"""Tests for MQTT source_type validation and fallback behavior."""

from unittest.mock import MagicMock, patch

import pytest

from cyberwave.constants import (
    SOURCE_TYPE_EDGE,
    SOURCE_TYPE_EDGE_LEADER,
    SOURCE_TYPE_EDIT,
    SOURCE_TYPE_TELE,
    SOURCE_TYPES,
)
from cyberwave.mqtt import CyberwaveMQTTClient


@pytest.fixture
def mqtt_client():
    with patch("cyberwave.mqtt.mqtt.Client"):
        yield CyberwaveMQTTClient(
            mqtt_broker="localhost",
            mqtt_port=1883,
            mqtt_username="user",
            api_key="api_key_secret",
            auto_connect=False,
            source_type=SOURCE_TYPE_TELE,
        )


def test_get_effective_source_type_prefers_explicit_value(mqtt_client):
    assert mqtt_client._get_effective_source_type(SOURCE_TYPE_EDIT) == SOURCE_TYPE_EDIT


def test_get_effective_source_type_falls_back_to_edge_when_not_set():
    with patch("cyberwave.mqtt.mqtt.Client"):
        client = CyberwaveMQTTClient(
            mqtt_broker="localhost",
            mqtt_port=1883,
            mqtt_username="user",
            api_key="api_key_secret",
            auto_connect=False,
            source_type=None,
        )

    assert client._get_effective_source_type(None) == SOURCE_TYPE_EDGE


def test_get_effective_source_type_rejects_invalid_values(mqtt_client):
    with pytest.raises(ValueError) as exc_info:
        mqtt_client._get_effective_source_type("invalid_source")

    error_message = str(exc_info.value)
    assert "Invalid source_type: invalid_source" in error_message
    for source_type in SOURCE_TYPES:
        assert source_type in error_message


def test_update_joint_state_uses_validated_source_type(mqtt_client):
    mqtt_client._handle_twin_update_with_telemetry = MagicMock()
    mqtt_client._is_rate_limited = MagicMock(return_value=False)
    mqtt_client.publish = MagicMock()

    mqtt_client.update_joint_state(
        twin_uuid="twin-uuid",
        joint_name="_1",
        position=0.75,
        timestamp=123.0,
        source_type=SOURCE_TYPE_EDGE_LEADER,
    )

    topic, message = mqtt_client.publish.call_args.args[:2]
    assert topic == "cyberwave/joint/twin-uuid/update"
    assert message["source_type"] == SOURCE_TYPE_EDGE_LEADER
    assert message["timestamp"] == 123.0


def test_update_joints_state_rejects_invalid_source_type_before_publish(mqtt_client):
    mqtt_client._handle_twin_update_with_telemetry = MagicMock()
    mqtt_client.publish = MagicMock()

    with pytest.raises(ValueError, match="Invalid source_type: nope"):
        mqtt_client.update_joints_state(
            twin_uuid="twin-uuid",
            joint_positions={"_1": 0.1},
            source_type="nope",
        )

    mqtt_client._handle_twin_update_with_telemetry.assert_not_called()
    mqtt_client.publish.assert_not_called()
