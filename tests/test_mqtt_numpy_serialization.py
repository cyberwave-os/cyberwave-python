"""Command arguments computed with numpy/OpenCV must reach the wire.

A distance measured from detected image corners is naturally a ``numpy.float32``;
passed to ``move_forward()`` it used to make ``json.dumps`` raise, so the command
was dropped with only "the MQTT client refused it" for the caller to go on.
"""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from cyberwave.mqtt import CyberwaveMQTTClient, _replace_non_finite


@pytest.fixture
def client():
    c = CyberwaveMQTTClient(mqtt_password="test-pw", auto_connect=False)
    c.connected = True
    c.client = MagicMock()
    c.client.publish.return_value.rc = 0
    return c


def test_publish_accepts_numpy_scalars_and_arrays(client):
    ok = client.publish(
        "t",
        {"data": {"distance": np.float32(2.25), "steps": np.int64(3), "xyz": np.array([0.5, 1.0])}},
    )
    assert ok is True
    data = json.loads(client.client.publish.call_args[0][1])["data"]
    assert data == {"distance": 2.25, "steps": 3, "xyz": [0.5, 1.0]}
    assert isinstance(data["steps"], int)


def test_publish_maps_non_finite_numpy_values_to_null(client):
    assert client.publish("t", {"a": np.float32("nan"), "b": np.array([1.0, np.inf])}) is True
    parsed = json.loads(
        client.client.publish.call_args[0][1],
        parse_constant=lambda token: pytest.fail(f"invalid JSON constant {token}"),
    )
    assert parsed["a"] is None and parsed["b"] == [1.0, None]


def test_replace_non_finite_handles_numpy():
    assert _replace_non_finite({"x": np.float64("inf"), "y": np.float32(1.5)}) == {"x": None, "y": 1.5}


def test_unknown_objects_are_still_rejected(client):
    assert client.publish("t", {"x": object()}) is False
