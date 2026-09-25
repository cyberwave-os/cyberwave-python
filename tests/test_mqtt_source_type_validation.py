"""Tests for MQTT source_type validation and fallback behavior."""

import logging
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


def test_update_joints_state_measured_uses_positions(mqtt_client):
    mqtt_client._handle_twin_update_with_telemetry = MagicMock()
    mqtt_client.publish = MagicMock()

    mqtt_client.update_joints_state(
        twin_uuid="twin-uuid",
        joint_positions={"_1": 0.5},
        source_type=SOURCE_TYPE_EDGE,
        velocities={"_1": 0.1},
        efforts={"_1": 0.2},
        timestamp=123.0,
    )

    _, message = mqtt_client.publish.call_args.args[:2]
    assert message["positions"] == {"_1": 0.5}
    assert message["velocities"] == {"_1": 0.1}
    assert message["efforts"] == {"_1": 0.2}
    assert "target_positions" not in message


def test_update_joints_state_as_targets_uses_target_fields(mqtt_client):
    mqtt_client._handle_twin_update_with_telemetry = MagicMock()
    mqtt_client.publish = MagicMock()

    mqtt_client.update_joints_state(
        twin_uuid="twin-uuid",
        joint_positions={"_1": 0.5, "_2": -0.3},
        source_type=SOURCE_TYPE_TELE,
        velocities={"_1": 0.0, "_2": 0.0},
        efforts={"_1": 0.0, "_2": 0.0},
        timestamp=123.0,
        as_targets=True,
    )

    topic, message = mqtt_client.publish.call_args.args[:2]
    assert topic == "cyberwave/joint/twin-uuid/update"
    assert message["source_type"] == SOURCE_TYPE_TELE
    assert message["target_positions"] == {"_1": 0.5, "_2": -0.3}
    assert message["target_velocities"] == {"_1": 0.0, "_2": 0.0}
    assert message["target_efforts"] == {"_1": 0.0, "_2": 0.0}
    # A command payload must never carry the measured field names.
    assert "positions" not in message
    assert "velocities" not in message
    assert "efforts" not in message


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


@pytest.mark.parametrize("channel", ["velocities", "efforts"])
def test_target_channel_can_be_sent_without_fabricated_positions(mqtt_client, channel):
    mqtt_client._handle_twin_update_with_telemetry = MagicMock()
    mqtt_client.publish = MagicMock()
    mqtt_client.update_joints_state(
        twin_uuid="twin-uuid",
        joint_positions={},
        as_targets=True,
        **{channel: {"wheel": 0.0}},
    )
    _, message = mqtt_client.publish.call_args.args[:2]
    assert message[f"target_{channel}"] == {"wheel": 0.0}
    assert set(message) == {"source_type", "timestamp", f"target_{channel}"}


@pytest.mark.parametrize("as_targets", [False, True])
def test_empty_joint_payload_is_rejected_before_telemetry(mqtt_client, as_targets):
    mqtt_client._handle_twin_update_with_telemetry = MagicMock()
    mqtt_client.publish = MagicMock()
    with pytest.raises(ValueError, match="cannot be empty"):
        mqtt_client.update_joints_state("twin-uuid", {}, as_targets=as_targets)
    mqtt_client.publish.assert_not_called()
    mqtt_client._handle_twin_update_with_telemetry.assert_not_called()


def test_measured_state_empty_position_contract_is_unchanged(mqtt_client):
    mqtt_client.publish = MagicMock()
    with pytest.raises(ValueError, match="cannot be empty"):
        mqtt_client.update_joints_state("twin-uuid", {}, efforts={"wheel": 0.5})
    mqtt_client.publish.assert_not_called()


@pytest.mark.parametrize("channel", ["positions", "velocities", "efforts"])
def test_target_only_client_does_not_own_plant_telemetry(mqtt_client, channel):
    """Stopping an external controller must not disconnect the plant's cameras."""
    mqtt_client.publish = MagicMock()
    positions = {"_1": 0.5} if channel == "positions" else {}
    extra = {} if channel == "positions" else {channel: {"_1": 0.5}}
    mqtt_client.update_joints_state("twin-uuid", positions, as_targets=True, **extra)

    assert mqtt_client.twin_uuids == []
    assert mqtt_client.twin_uuids_with_telemetry_start == []
    mqtt_client.disconnect()
    mqtt_client.publish.assert_called_once()
    topic, message = mqtt_client.publish.call_args.args[:2]
    assert topic == "cyberwave/joint/twin-uuid/update"
    assert message[f"target_{channel}"] == {"_1": 0.5}


def test_targets_preserve_existing_measured_producer_lifecycle(mqtt_client):
    """A client that actually produces state still owns its normal lifecycle."""
    mqtt_client.publish = MagicMock()
    mqtt_client.update_joints_state("twin-uuid", {"_1": 0.2})
    mqtt_client.update_joints_state("twin-uuid", {"_1": 0.5}, as_targets=True)
    assert mqtt_client.twin_uuids == ["twin-uuid"]
    assert mqtt_client.twin_uuids_with_telemetry_start == ["twin-uuid"]
    mqtt_client.disconnect()
    types = [call.args[1].get("type") for call in mqtt_client.publish.call_args_list]
    assert types.count("connected") == 1
    assert types.count("telemetry_start") == 1
    assert types.count("disconnected") == 1
    assert types.count("telemetry_end") == 1


def test_publish_success_does_not_emit_debug_publish_noise(mqtt_client, caplog):
    mqtt_client.connected = True
    mqtt_client.client.publish = MagicMock(return_value=MagicMock(rc=0))
    caplog.set_level(logging.DEBUG, logger="cyberwave.mqtt")

    mqtt_client.publish("cyberwave/twin/test/driverlog", {"message": "ok"})

    assert not any(
        "Published to cyberwave/twin/test/driverlog" in record.getMessage()
        for record in caplog.records
    )
