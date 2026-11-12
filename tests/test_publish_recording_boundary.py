"""``publish_recording_boundary`` opens or closes one recording window.

A simulator's recording toggle uses it: turning recording off must close the
window without reopening it, and neither half may tear the live streams down.
"""

from unittest.mock import MagicMock, patch

import pytest

from cyberwave.constants import SOURCE_TYPE_SIM
from cyberwave.mqtt import RECORDING_BOUNDARY_KEY, CyberwaveMQTTClient


@pytest.fixture
def mqtt_client():
    with patch("cyberwave.mqtt.mqtt.Client"):
        yield CyberwaveMQTTClient(
            mqtt_broker="localhost",
            mqtt_port=1883,
            mqtt_username="user",
            api_key="api_key_secret",
            auto_connect=False,
            source_type=SOURCE_TYPE_SIM,
            announce_twin_sessions=False,
        )


def test_start_carries_identity_marker_and_observations(mqtt_client):
    mqtt_client.publish = MagicMock()

    mqtt_client.publish_recording_boundary(
        "twin-abc",
        "start",
        stream_source="simulation",
        stream_instance_id="workload-1",
        observations={"joint_1": 0.4},
    )

    (topic, message), _ = mqtt_client.publish.call_args
    assert topic.endswith("cyberwave/twin/twin-abc/telemetry")
    assert message["type"] == "telemetry_start"
    assert message[RECORDING_BOUNDARY_KEY] is True
    assert message["stream_source"] == "simulation"
    assert message["stream_instance_id"] == "workload-1"
    assert message["source_type"] == SOURCE_TYPE_SIM
    assert message["observations"] == {"joint_1": 0.4}


def test_end_is_a_single_message_with_no_reopen(mqtt_client):
    mqtt_client.publish = MagicMock()

    mqtt_client.publish_recording_boundary(
        "twin-abc",
        "end",
        stream_source="simulation",
        stream_instance_id="workload-1",
        observations={"joint_1": 0.4},
    )

    assert mqtt_client.publish.call_count == 1
    (_topic, message), _ = mqtt_client.publish.call_args
    assert message["type"] == "telemetry_end"
    assert message[RECORDING_BOUNDARY_KEY] is True
    # A window's end carries no starting state.
    assert "observations" not in message
    # No sensor: this closes the window, not one camera's stream.
    assert "sensor" not in message


def test_does_not_touch_session_tracking(mqtt_client):
    mqtt_client.publish = MagicMock()

    mqtt_client.publish_recording_boundary(
        "twin-abc", "start", stream_source="simulation", stream_instance_id="w"
    )

    assert "twin-abc" not in mqtt_client.twin_uuids_with_telemetry_start


def test_rejects_unknown_state(mqtt_client):
    with pytest.raises(ValueError):
        mqtt_client.publish_recording_boundary(
            "twin-abc", "pause", stream_source="simulation", stream_instance_id="w"
        )
