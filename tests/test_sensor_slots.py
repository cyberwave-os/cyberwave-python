"""Sensor slots: ``publish_readings`` (live values) and the twin's own sensors.

``publish_readings`` fills ``metrics.readings`` -- the bag the backend stores as
a slot's live value and judges against its range -- unlike ``publish_metrics``,
whose free-form channels become metric rows. The sensor helpers call the
hand-written ``/api/v1/twins/{uuid}/sensors`` routes.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import paho.mqtt.client as mqtt
import pytest

from cyberwave.mqtt import MAX_METRIC_STRING_LENGTH, CyberwaveMQTTClient
from cyberwave.resources import TwinManager

TWIN = "8f1c2d3e-0000-4000-8000-000000000001"
NOW = 1788433200.25


@pytest.fixture
def mqtt_client() -> Any:
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = mqtt_client_cls.return_value
        mqtt_client.publish.return_value = MagicMock(rc=mqtt.MQTT_ERR_SUCCESS)
        yield mqtt_client


def _client() -> CyberwaveMQTTClient:
    client = CyberwaveMQTTClient(
        mqtt_broker="localhost",
        mqtt_port=1883,
        mqtt_username="user",
        api_key="api_key_secret",
        auto_connect=False,
        client_id="edge-driver-01",
    )
    client.connected = True
    return client


def _sent(mqtt_client: MagicMock) -> list[tuple[str, dict[str, Any]]]:
    return [
        (call.args[0], json.loads(call.args[1]))
        for call in mqtt_client.publish.call_args_list
    ]


# -- publish_readings -----------------------------------------------------------


def test_readings_go_to_the_metrics_topic_in_the_readings_bag(mqtt_client):
    client = _client()

    assert client.publish_readings(
        TWIN,
        {"value": 21.5, "humidity": {"value": 40, "unit": "%", "at": NOW - 5}},
        timestamp=NOW,
    )

    [(topic, message)] = _sent(mqtt_client)
    assert topic == f"{client.topic_prefix}cyberwave/twin/{TWIN}/metrics"
    assert message["timestamp"] == NOW
    assert message["metrics"] == {
        "readings": {
            # No unit: the sensor's declared unit applies.
            "value": {"value": 21.5, "at": NOW},
            "humidity": {"value": 40.0, "unit": "%", "at": NOW - 5},
        }
    }


def test_an_empty_unit_is_left_out(mqtt_client):
    _client().publish_readings(TWIN, {"value": {"value": 1, "unit": " "}}, timestamp=NOW)

    [(_, message)] = _sent(mqtt_client)
    assert message["metrics"]["readings"]["value"] == {"value": 1.0, "at": NOW}


@pytest.mark.parametrize(
    ("readings", "error"),
    [
        ({}, ValueError),
        ({"": 1.0}, ValueError),
        ({"value": {"unit": "°C"}}, ValueError),
        ({"value": {"value": 1, "channel": "x"}}, ValueError),
        ({"value": float("nan")}, ValueError),
        ({"value": "21.5"}, TypeError),
        ({"value": "warm"}, TypeError),
        ({"value": "w" * (MAX_METRIC_STRING_LENGTH + 1)}, TypeError),
        ({"value": {"value": 1, "unit": 5}}, TypeError),
        ({"value": {"value": 1, "at": NOW * 1000}}, ValueError),
    ],
)
def test_readings_the_backend_would_drop_are_refused(mqtt_client, readings, error):
    with pytest.raises(error):
        _client().publish_readings(TWIN, readings, timestamp=NOW)

    assert _sent(mqtt_client) == []


# -- the twin's own sensors -------------------------------------------------------


def _manager(result: Any) -> tuple[TwinManager, Mock]:
    api_client = Mock()
    api_client.param_serialize.return_value = ("serialized",)
    api_client.response_deserialize.return_value = SimpleNamespace(data=result)
    return TwinManager(SimpleNamespace(api_client=api_client)), api_client


def _request(api_client: Mock) -> dict[str, Any]:
    return api_client.param_serialize.call_args.kwargs


def test_add_sensor_posts_the_row_with_pose_and_parameters():
    stored = {"name": "probe", "type": "temperature"}
    manager, api_client = _manager({"sensor": stored})

    result = manager.add_sensor(
        TWIN,
        "probe",
        "temperature",
        parent_link="base",
        position=(0.1, 0, 0.2),
        unit="°C",
        operating_range=(0, None),
    )

    assert result == stored
    request = _request(api_client)
    assert (request["method"], request["resource_path"]) == (
        "POST",
        "/api/v1/twins/{uuid}/sensors",
    )
    assert request["path_params"] == {"uuid": TWIN}
    assert request["body"] == {
        "sensor": {
            "name": "probe",
            "type": "temperature",
            "parent_link": "base",
            "parameters": {"unit": "°C", "operating_range": {"min": 0, "max": None}},
            "pose": {
                "position": {"x": 0.1, "y": 0.0, "z": 0.2},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        }
    }


def test_add_sensor_leaves_out_what_was_not_given():
    manager, api_client = _manager({"sensor": {}})

    manager.add_sensor(TWIN, "value", "generic")

    assert _request(api_client)["body"] == {
        "sensor": {"name": "value", "type": "generic", "parameters": {}}
    }


def test_set_sensor_unit_is_a_merge_patch_on_the_one_sensor():
    manager, api_client = _manager({"sensor": {"name": "value"}})

    manager.set_sensor_unit(TWIN, "value", "°C")
    request = _request(api_client)
    assert (request["method"], request["resource_path"]) == (
        "PATCH",
        "/api/v1/twins/{uuid}/sensors/{name}",
    )
    assert request["path_params"] == {"uuid": TWIN, "name": "value"}
    assert request["body"] == {"patch": {"parameters": {"unit": "°C"}}}

    # None removes the unit (RFC 7386).
    manager.set_sensor_unit(TWIN, "value", None)
    assert _request(api_client)["body"] == {"patch": {"parameters": {"unit": None}}}


def test_list_and_remove_sensor_call_their_routes():
    rows = [{"name": "cam", "origin": "asset"}, {"name": "probe", "origin": "twin"}]
    manager, api_client = _manager({"sensors": rows, "schema_hash": "h"})

    assert manager.list_sensors(TWIN) == rows
    assert _request(api_client)["method"] == "GET"

    manager.remove_sensor(TWIN, "probe")
    request = _request(api_client)
    assert (request["method"], request["path_params"]) == (
        "DELETE",
        {"uuid": TWIN, "name": "probe"},
    )
