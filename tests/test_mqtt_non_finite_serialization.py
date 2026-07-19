"""The MQTT wire must always be valid JSON, even when a producer hands the
client non-finite floats (``NaN`` / ``inf``).

A robot driver legitimately ends up with an unmeasured joint value (e.g. Gazebo
reports ``NaN`` effort for a passive gripper joint). ``json.dumps`` defaults to
``allow_nan=True`` and emits the bare ``NaN`` token, which is invalid JSON — a
strict consumer (the browser's ``JSON.parse``, the C++/other SDKs) then rejects
the ENTIRE payload and joint/pose rendering silently stops. These tests lock the
client's guarantee that non-finite values are serialized as ``null`` instead.
"""

import json
from unittest.mock import MagicMock

import pytest

from cyberwave.mqtt import CyberwaveMQTTClient, _replace_non_finite


def _strict_loads(payload: str):
    """Parse like a strict (JS ``JSON.parse``-style) consumer: reject NaN/inf."""

    def _reject(token):
        raise ValueError(f"invalid JSON constant: {token}")

    return json.loads(payload, parse_constant=_reject)


@pytest.fixture
def client():
    c = CyberwaveMQTTClient(mqtt_password="test-pw", auto_connect=False)
    c.connected = True
    c.client = MagicMock()
    return c


def test_publish_serializes_nan_as_null(client):
    client.publish(
        "localcyberwave/joint/twin-1/update",
        {
            "positions": {"joint_1": 0.5},
            "efforts": {"joint_1": 12.3, "gripper": float("nan")},
        },
    )

    payload = client.client.publish.call_args[0][1]
    # Must be valid JSON for a strict consumer (would previously raise on NaN).
    parsed = _strict_loads(payload)
    assert parsed["efforts"]["gripper"] is None
    assert parsed["efforts"]["joint_1"] == 12.3
    assert parsed["positions"]["joint_1"] == 0.5


def test_publish_serializes_infinity_as_null(client):
    client.publish(
        "t",
        {"a": float("inf"), "b": float("-inf"), "nested": [1.0, float("nan")]},
    )
    parsed = _strict_loads(client.client.publish.call_args[0][1])
    assert parsed["a"] is None
    assert parsed["b"] is None
    assert parsed["nested"] == [1.0, None]


def test_publish_leaves_finite_values_untouched(client):
    client.publish("t", {"x": 1.5, "y": -2, "s": "ok", "flag": True, "none": None})
    parsed = _strict_loads(client.client.publish.call_args[0][1])
    assert parsed["x"] == 1.5
    assert parsed["y"] == -2
    assert parsed["s"] == "ok"
    assert parsed["flag"] is True
    assert parsed["none"] is None


def test_replace_non_finite_pure_helper():
    out = _replace_non_finite(
        {"a": float("nan"), "b": [float("inf"), 3.0], "c": {"d": 1.0}}
    )
    assert out == {"a": None, "b": [None, 3.0], "c": {"d": 1.0}}
