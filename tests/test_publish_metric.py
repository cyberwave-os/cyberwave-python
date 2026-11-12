"""``publish_metric`` / ``publish_metrics``: structured float/int/bool metrics.

The backend reads ``twin/{uuid}/metrics/a/b`` as channel ``a.b`` -- the
channel the bag key ``"a.b"`` names -- and drops any sample that is not one
finite float, int or bool. So these pin what the backend cannot fix up if the
SDK gets it wrong: the topic path, the one-``value`` body, and refusing a
sample the backend would drop before it is ever sent.
"""

from __future__ import annotations

import json
import math
from enum import IntEnum
from fractions import Fraction
from typing import Any
from unittest.mock import MagicMock, patch

import paho.mqtt.client as mqtt
import pytest

from cyberwave.config import CyberwaveConfig
from cyberwave.mqtt import MAX_METRIC_CHANNEL_LENGTH, CyberwaveMQTTClient
from cyberwave.mqtt_client import CyberwaveMQTTClient as WrapperMQTTClient

TWIN = "8f1c2d3e-0000-4000-8000-000000000001"
NOW = 1788433200.25


@pytest.fixture
def mqtt_client() -> Any:
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = mqtt_client_cls.return_value
        mqtt_client.publish.return_value = MagicMock(rc=mqtt.MQTT_ERR_SUCCESS)
        yield mqtt_client


def _client(**kwargs: Any) -> CyberwaveMQTTClient:
    client = CyberwaveMQTTClient(
        mqtt_broker="localhost",
        mqtt_port=1883,
        mqtt_username="user",
        api_key="api_key_secret",
        auto_connect=False,
        client_id="edge-driver-01",
        **kwargs,
    )
    client.connected = True
    return client


def _sent(mqtt_client: MagicMock) -> list[tuple[str, dict[str, Any]]]:
    return [
        (call.args[0], json.loads(call.args[1]))
        for call in mqtt_client.publish.call_args_list
    ]


# -- publish_metric: topic ------------------------------------------------------


@pytest.mark.parametrize(
    ("channel", "path"),
    [
        ("cycle_count", "cycle_count"),
        ("arm.gripper", "arm/gripper"),
        ("arm.gripper.left", "arm/gripper/left"),
        ("arm.source_type", "arm/source_type"),
        ("Motor Temp", "Motor Temp"),
        ("a" * MAX_METRIC_CHANNEL_LENGTH, "a" * MAX_METRIC_CHANNEL_LENGTH),
    ],
)
def test_the_channel_dots_are_the_topic_levels(
    mqtt_client: MagicMock, channel: str, path: str
) -> None:
    _client().publish_metric(TWIN, channel, 1.0)

    [(topic, _)] = _sent(mqtt_client)
    assert topic == f"cyberwave/twin/{TWIN}/metrics/{path}"


def test_the_topic_prefix_applies(mqtt_client: MagicMock) -> None:
    _client(topic_prefix="staging/").publish_metric(TWIN, "arm.gripper", 1.0)

    [(topic, _)] = _sent(mqtt_client)
    assert topic == f"staging/cyberwave/twin/{TWIN}/metrics/arm/gripper"


@pytest.mark.parametrize(
    "channel",
    [
        "",
        ".arm",
        "arm.",
        "arm..gripper",
        "arm/gripper",
        "arm.+",
        "arm.#",
        "a\x00b",
        "a" * (MAX_METRIC_CHANNEL_LENGTH + 1),
        "source_type",
        None,
    ],
)
def test_a_channel_that_names_no_topic_is_refused_before_publishing(
    mqtt_client: MagicMock, channel: Any
) -> None:
    with pytest.raises(ValueError):
        _client().publish_metric(TWIN, channel, 1.0)
    assert not mqtt_client.publish.called


# -- publish_metric: the sample ---------------------------------------------------


@pytest.mark.parametrize("value", [12.5, -3, 0, True, False, 2**63 - 1, -(2**63)])
def test_a_scalar_is_the_one_value_beside_the_envelope(
    mqtt_client: MagicMock, value: Any
) -> None:
    _client().publish_metric(TWIN, "arm.gripper", value, timestamp=NOW)

    [(_, message)] = _sent(mqtt_client)
    # ``session_id`` is stamped by ``publish`` itself; the backend reads it as
    # envelope, so it is not a second field beside ``value``.
    assert message == {
        "source_type": "edge",
        "timestamp": NOW,
        "value": value,
        "session_id": "edge-driver-01",
    }
    assert type(message["value"]) is type(value)


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ("12.5", TypeError),
        (None, TypeError),
        ([1, 2], TypeError),
        ({"force_n": 1.0}, TypeError),
        (math.nan, ValueError),
        (math.inf, ValueError),
        (2**63, ValueError),
        (-(2**63) - 1, ValueError),
        # Past 2**63 the backend's history cannot tell a float from an int
        # that overflowed, so it refuses both.
        (2.0**63, ValueError),
        (-(2.0**63), ValueError),
    ],
)
def test_anything_but_a_finite_scalar_is_refused_before_publishing(
    mqtt_client: MagicMock, value: Any, error: type[Exception]
) -> None:
    with pytest.raises(error, match="arm.gripper"):
        _client().publish_metric(TWIN, "arm.gripper", value)
    assert not mqtt_client.publish.called


@pytest.mark.parametrize(
    ("value", "sent"),
    [
        (Fraction(1, 4), 0.25),
        (IntEnum("Gear", "LOW HIGH").HIGH, 2),
    ],
)
def test_another_real_number_is_sent_as_the_builtin(
    mqtt_client: MagicMock, value: Any, sent: Any
) -> None:
    _client().publish_metric(TWIN, "n", value)

    [(_, message)] = _sent(mqtt_client)
    assert message["value"] == sent
    assert type(message["value"]) is type(sent)


def test_a_numpy_scalar_is_sent_as_the_builtin(mqtt_client: MagicMock) -> None:
    """A driver reading a sensor through numpy should not have to unwrap."""
    np = pytest.importorskip("numpy")
    client = _client()

    client.publish_metric(TWIN, "a", np.float32(1.5))
    client.publish_metric(TWIN, "b", np.int64(-7))
    client.publish_metric(TWIN, "c", np.bool_(True))
    client.publish_metrics(TWIN, {"d": np.uint8(3), "e": np.float64(2.5)})

    values = [
        message.get("value", message.get("metrics"))
        for _, message in _sent(mqtt_client)
    ]
    assert values == [1.5, -7, True, {"d": 3, "e": 2.5}]
    assert [type(v) for v in values[:3]] == [float, int, bool]
    with pytest.raises(ValueError, match="finite"):
        client.publish_metric(TWIN, "f", np.float32("nan"))
    with pytest.raises(ValueError, match="64 bits"):
        client.publish_metric(TWIN, "g", np.uint64(2**64 - 1))
    with pytest.raises(TypeError):
        client.publish_metric(TWIN, "h", np.array([1.0, 2.0]))


def test_the_timestamp_defaults_to_now(mqtt_client: MagicMock) -> None:
    with patch("cyberwave.mqtt.time.time", return_value=NOW):
        _client().publish_metric(TWIN, "cycle_count", 1)

    [(_, message)] = _sent(mqtt_client)
    assert message["timestamp"] == NOW


def test_an_explicit_timestamp_is_kept(mqtt_client: MagicMock) -> None:
    _client().publish_metric(TWIN, "cycle_count", 1, timestamp=NOW - 5)
    _client().publish_metrics(TWIN, {"cycle_count": 1}, timestamp=1788433200)

    assert [message["timestamp"] for _, message in _sent(mqtt_client)] == [
        NOW - 5,
        1788433200.0,
    ]


@pytest.mark.parametrize(
    ("timestamp", "error"),
    [
        pytest.param(0.0, ValueError, id="zero"),
        pytest.param(NOW * 1000, ValueError, id="milliseconds"),
        pytest.param(NOW + 11 * 365 * 86400, ValueError, id="past-ten-years-ahead"),
        pytest.param(math.nan, ValueError, id="nan"),
        pytest.param(math.inf, ValueError, id="inf"),
        pytest.param("1788433200", TypeError, id="string"),
        pytest.param(True, TypeError, id="bool"),
    ],
)
def test_a_timestamp_that_is_not_unix_seconds_is_refused(
    mqtt_client: MagicMock, timestamp: Any, error: type[Exception]
) -> None:
    """The backend would keep the sample but stamp its history with the time it
    arrived, so a caller passing milliseconds would never find out."""
    with pytest.raises(error):
        _client().publish_metric(TWIN, "cycle_count", 1, timestamp=timestamp)
    with pytest.raises(error):
        _client().publish_metrics(TWIN, {"cycle_count": 1}, timestamp=timestamp)
    assert not mqtt_client.publish.called


def test_source_type_follows_the_client_and_can_be_overridden(
    mqtt_client: MagicMock,
) -> None:
    client = _client(source_type="sim")
    client.publish_metric(TWIN, "cycle_count", 1)
    client.publish_metric(TWIN, "cycle_count", 1, source_type="edge")

    assert [message["source_type"] for _, message in _sent(mqtt_client)] == [
        "sim",
        "edge",
    ]


@pytest.mark.parametrize("rc", [mqtt.MQTT_ERR_SUCCESS, mqtt.MQTT_ERR_NO_CONN])
def test_it_returns_whether_the_client_accepted_it(
    mqtt_client: MagicMock, rc: int
) -> None:
    mqtt_client.publish.return_value = MagicMock(rc=rc)
    assert _client().publish_metric(TWIN, "cycle_count", 1) is (
        rc == mqtt.MQTT_ERR_SUCCESS
    )


# -- publish_metrics: the bag ---------------------------------------------------


def test_the_bag_nests_its_samples_under_metrics(mqtt_client: MagicMock) -> None:
    _client().publish_metrics(
        TWIN, {"arm.gripper": 12.5, "motor_on": True, "faults": 0}, timestamp=NOW
    )

    [(topic, message)] = _sent(mqtt_client)
    assert topic == f"cyberwave/twin/{TWIN}/metrics"
    assert message == {
        "source_type": "edge",
        "timestamp": NOW,
        "metrics": {"arm.gripper": 12.5, "motor_on": True, "faults": 0},
        "session_id": "edge-driver-01",
    }


@pytest.mark.parametrize(
    "metrics",
    [
        {},
        {"": 1},
        {"n": "1"},
        {"n": {"force_n": 1.0}},
        {"n": math.nan},
        # Named as ``publish_metric`` names a channel, or not at all.
        {"arm/gripper": 1},
        {"arm.+": 1},
        {"arm.#": 1},
        {"arm..gripper": 1},
        {"a" * (MAX_METRIC_CHANNEL_LENGTH + 1): 1},
        {1: 1},
    ],
)
def test_a_bag_the_backend_would_not_read_as_metrics_is_refused(
    mqtt_client: MagicMock, metrics: dict[str, Any]
) -> None:
    with pytest.raises((ValueError, TypeError)):
        _client().publish_metrics(TWIN, metrics)
    assert not mqtt_client.publish.called


def test_the_callers_bag_is_not_modified(mqtt_client: MagicMock) -> None:
    bag = {"n": 1}
    _client().publish_metrics(TWIN, bag)
    assert bag == {"n": 1}


# -- the compatibility wrapper ------------------------------------------------------


def _wrapper() -> tuple[WrapperMQTTClient, MagicMock]:
    config = CyberwaveConfig(api_key="api_key_secret", mqtt_username="user")
    with patch("cyberwave.mqtt_client.BaseMQTTClient") as base_cls:
        wrapper = WrapperMQTTClient(config=config)
    return wrapper, base_cls.return_value


def test_the_wrapper_forwards_publish_metric() -> None:
    wrapper, base = _wrapper()
    base.publish_metric.return_value = True

    assert (
        wrapper.publish_metric(
            TWIN, "arm.gripper", 1.0, timestamp=NOW, source_type="sim"
        )
        is True
    )
    base.publish_metric.assert_called_once_with(
        TWIN, "arm.gripper", 1.0, timestamp=NOW, source_type="sim"
    )


def test_the_wrapper_forwards_publish_metrics() -> None:
    wrapper, base = _wrapper()
    base.publish_metrics.return_value = False

    assert wrapper.publish_metrics(TWIN, {"n": 1}, timestamp=NOW) is False
    base.publish_metrics.assert_called_once_with(
        TWIN, {"n": 1}, timestamp=NOW, source_type=None
    )
