"""A relay client must not end the telemetry session of the twin it relays for.

A ``telemetry_end`` without a sensor makes the media service finalize the whole
twin: every camera producer and recording. A navigation bridge that published
the twin's joints and then shut down did exactly that to the simulator's (or
the robot driver's) cameras, which were still streaming.
"""

from unittest.mock import MagicMock, patch

import pytest

from cyberwave.mqtt import CyberwaveMQTTClient


def _client(**kwargs):
    with patch("cyberwave.mqtt.mqtt.Client"):
        client = CyberwaveMQTTClient(
            mqtt_broker="localhost",
            mqtt_port=1883,
            api_key="api_key_secret",
            auto_connect=False,
            **kwargs,
        )
    client.connected = True
    client.publish = MagicMock(return_value=True)
    return client


def _types(client):
    return [call.args[1].get("type") for call in client.publish.call_args_list]


def _lifecycle(client):
    return [
        t
        for t in _types(client)
        if t in {"connected", "telemetry_start", "disconnected", "telemetry_end"}
    ]


@pytest.mark.parametrize("announce", [True, False])
def test_publishing_state_opens_and_disconnect_closes_only_when_announcing(announce):
    client = _client(announce_twin_sessions=announce)

    client.update_joints_state("twin-a", {"j1": 0.1}, timestamp=1.0)
    client.disconnect()

    if announce:
        assert _lifecycle(client) == [
            "connected",
            "telemetry_start",
            "disconnected",
            "telemetry_end",
        ]
    else:
        assert _lifecycle(client) == []
    # The state update itself goes out either way.
    assert len(_types(client)) > len(_lifecycle(client))


def test_an_explicit_start_is_still_announced_and_ended():
    client = _client(announce_twin_sessions=False)

    client.update_joints_state("relayed", {"j1": 0.1}, timestamp=1.0)
    client.publish_telemetry_start("owned")
    client.disconnect()

    published = [
        (call.args[1].get("type"), call.args[0].split("/")[-2])
        for call in client.publish.call_args_list
        if call.args[1].get("type")
        in {"connected", "telemetry_start", "disconnected", "telemetry_end"}
    ]
    assert published == [
        ("connected", "owned"),
        ("telemetry_start", "owned"),
        ("disconnected", "owned"),
        ("telemetry_end", "owned"),
    ]


def test_reconnect_does_not_announce_relayed_twins():
    client = _client(announce_twin_sessions=False, twin_uuids=["relayed"])
    client.connected = False

    def connect(*_args, **_kwargs):
        client.connected = True

    client.client.connect.side_effect = connect
    client.connect()

    assert _lifecycle(client) == []


def test_every_stream_of_a_twin_gets_its_own_end():
    client = _client()
    client.publish_telemetry_start("spot")

    for sensor in ("frontleft_fisheye_image", "frontleft_depth_in_visual_frame"):
        client.publish_telemetry_end(
            "spot",
            sensor=sensor,
            stream_source="simulation",
            stream_instance_id="workload-1",
        )
    client.disconnect()

    ends = [
        call.args[1]
        for call in client.publish.call_args_list
        if call.args[1].get("type") == "telemetry_end"
    ]
    # One end per stream. The stream ends leave the twin's session open, so
    # disconnect() still closes it with one twin-wide end.
    assert [end.get("sensor") for end in ends] == [
        "frontleft_fisheye_image",
        "frontleft_depth_in_visual_frame",
        None,
    ]
    assert all(end["stream_instance_id"] == "workload-1" for end in ends[:2])


def test_a_stream_end_leaves_the_twin_session_open():
    client = _client()
    client.update_joints_state("spot", {"j1": 0.1}, timestamp=1.0)

    client.publish_telemetry_end("spot", sensor="frontleft_fisheye_image")
    # The twin keeps publishing state: that must not reopen its session.
    client.update_joints_state("spot", {"j1": 0.2}, timestamp=2.0)
    client.disconnect()

    assert _lifecycle(client) == [
        "connected",
        "telemetry_start",
        "telemetry_end",  # the stream's
        "disconnected",
        "telemetry_end",  # the twin's
    ]


def test_a_twin_wide_end_is_still_sent_once():
    client = _client()
    client.publish_telemetry_start("arm")

    client.publish_telemetry_end("arm")
    client.publish_telemetry_end("arm")
    client.disconnect()

    assert _lifecycle(client).count("telemetry_end") == 1
