"""``publish_telemetry_cut`` emits an unconditional telemetry_end + telemetry_start.

The recorder workflow node uses this to cut a bounded recording window into a
telemetry session it does not own (a driver's continuous stream): the end
bounds the window precisely, the reopen keeps the driver's stream recording.
"""

from unittest.mock import MagicMock, patch

import pytest

from cyberwave.constants import SOURCE_TYPE_TELE
from cyberwave.mqtt import (
    RECORDING_BOUNDARY_KEY,
    TELEMETRY_CUT_SENDER,
    TELEMETRY_CUT_SOURCE_SUBTYPE,
    CyberwaveMQTTClient,
)


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


def test_publish_telemetry_cut_emits_end_then_start(mqtt_client):
    mqtt_client.publish = MagicMock()

    mqtt_client.publish_telemetry_cut("twin-abc", source_type=SOURCE_TYPE_TELE)

    assert mqtt_client.publish.call_count == 2
    (end_topic, end_msg), _ = mqtt_client.publish.call_args_list[0]
    (start_topic, start_msg), _ = mqtt_client.publish.call_args_list[1]

    # Both land on the twin's telemetry topic.
    for topic in (end_topic, start_topic):
        assert "twin-abc" in topic
        assert topic.endswith("/telemetry")

    # End first, start second (reopen), start sorts strictly after the end.
    assert end_msg["type"] == "telemetry_end"
    assert start_msg["type"] == "telemetry_start"
    assert start_msg["timestamp"] > end_msg["timestamp"]
    assert end_msg["source_type"] == SOURCE_TYPE_TELE
    assert start_msg["source_type"] == SOURCE_TYPE_TELE


def test_publish_telemetry_cut_is_unconditional(mqtt_client):
    """Unlike ``publish_telemetry_end`` it must NOT depend on this client
    having previously tracked a telemetry_start for the twin — the recorder
    never sent the owning start."""
    mqtt_client.publish = MagicMock()

    # Twin was never registered / started on this client.
    assert "twin-xyz" not in mqtt_client.twin_uuids_with_telemetry_start

    mqtt_client.publish_telemetry_cut("twin-xyz")

    assert mqtt_client.publish.call_count == 2
    types = [msg["type"] for (_topic, msg), _ in mqtt_client.publish.call_args_list]
    assert types == ["telemetry_end", "telemetry_start"]


def test_publish_telemetry_cut_omits_source_type_when_unset(mqtt_client):
    mqtt_client.publish = MagicMock()

    mqtt_client.publish_telemetry_cut("twin-abc")

    for (_topic, msg), _ in mqtt_client.publish.call_args_list:
        assert "source_type" not in msg


def test_publish_telemetry_cut_marks_both_halves_as_recording_boundary(mqtt_client):
    """The peer stays connected across a cut. Without this marker
    media-service reads the ``telemetry_end`` as a disconnect and drops the
    twin's producers/consumers, killing the stream being recorded."""
    mqtt_client.publish = MagicMock()

    mqtt_client.publish_telemetry_cut("twin-abc", source_type=SOURCE_TYPE_TELE)

    assert mqtt_client.publish.call_count == 2
    for (_topic, msg), _ in mqtt_client.publish.call_args_list:
        assert msg[RECORDING_BOUNDARY_KEY] is True
        assert msg["sender"] == TELEMETRY_CUT_SENDER == "workflow"
        assert msg["source_subtype"] == TELEMETRY_CUT_SOURCE_SUBTYPE == "node_recorder"


def test_publish_telemetry_cut_attribution_is_overridable(mqtt_client):
    mqtt_client.publish = MagicMock()

    mqtt_client.publish_telemetry_cut(
        "twin-abc", sender="workflow", source_subtype="node_recorder_autostop"
    )

    for (_topic, msg), _ in mqtt_client.publish.call_args_list:
        assert msg["source_subtype"] == "node_recorder_autostop"
        # The behavioral flag is an invariant of the helper, not a caller knob.
        assert msg[RECORDING_BOUNDARY_KEY] is True


def test_publish_telemetry_end_is_not_marked_a_recording_boundary(mqtt_client):
    """A genuine end of telemetry must keep its original meaning: the owning
    publisher is going away and consumers SHOULD tear the session down."""
    mqtt_client.publish = MagicMock()
    mqtt_client.twin_uuids_with_telemetry_start.append("twin-abc")

    mqtt_client.publish_telemetry_end("twin-abc", sensor="wrist_camera")

    assert mqtt_client.publish.call_count == 1
    (_topic, msg), _ = mqtt_client.publish.call_args_list[0]
    assert msg["type"] == "telemetry_end"
    assert RECORDING_BOUNDARY_KEY not in msg
    assert "sender" not in msg
