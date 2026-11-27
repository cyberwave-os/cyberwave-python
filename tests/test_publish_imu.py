"""``publish_imu`` against the imu contract.

The golden fixture in ``common/protobuf/fixtures/mqtt/twin/v1/imu.json`` is the
contract's payload, so the SDK has to put the same topic and field names on the
wire, and refuse what Vector would drop or mark invalid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import paho.mqtt.client as mqtt
import pytest

from cyberwave.driver.interface.args import TopicSpec
from cyberwave.driver.interface.registry import resolve_topic_path
from cyberwave.manifest.driver_config import TWIN_IMU_TOPIC_SLUG
from cyberwave.mqtt import CyberwaveMQTTClient
from cyberwave.mqtt_client import CyberwaveMQTTClient as WrapperMQTTClient

_FIXTURE = (
    Path(__file__).resolve().parents[3] / "common/protobuf/fixtures/mqtt/twin/v1/imu.json"
)

TWIN = "8f1c2d3e-0000-4000-8000-000000000001"
GYRO = {"x": 0.01, "y": -0.02, "z": 0.35}
ACCEL = {"x": 0.12, "y": -0.05, "z": 9.79}
QUAT = {"w": 0.99875, "x": 0.0, "y": 0.0, "z": 0.04998}


@pytest.fixture
def paho() -> Any:
    with patch("cyberwave.mqtt.mqtt.Client") as client_cls:
        paho = client_cls.return_value
        paho.publish.return_value = MagicMock(rc=mqtt.MQTT_ERR_SUCCESS)
        yield paho


@pytest.fixture
def client(paho: MagicMock) -> CyberwaveMQTTClient:
    client = CyberwaveMQTTClient(
        mqtt_broker="localhost",
        mqtt_port=1883,
        mqtt_username="user",
        api_key="api_key_secret",
        auto_connect=False,
        source_type="edge",
    )
    client.connected = True
    return client


def _sent(paho: MagicMock) -> tuple[str, dict[str, Any]]:
    (call,) = paho.publish.call_args_list
    return call.args[0], json.loads(call.args[1])


def test_publishes_the_sample_on_the_twin_imu_topic(
    client: CyberwaveMQTTClient, paho: MagicMock
) -> None:
    assert (
        client.publish_imu(
            TWIN, angular_velocity=GYRO, linear_acceleration=ACCEL, sensor_id="imu"
        )
        is True
    )

    topic, message = _sent(paho)
    assert topic == f"cyberwave/twin/{TWIN}/imu"
    assert topic == TWIN_IMU_TOPIC_SLUG.format(twin_uuid=TWIN)
    assert message["source_type"] == "edge"
    assert message["sensor_id"] == "imu"
    assert message["angular_velocity"] == GYRO
    assert message["linear_acceleration"] == ACCEL
    assert isinstance(message["timestamp"], float)
    # No orientation, so no frame to name and no legacy aliases.
    assert not {"orientation", "coordinate_frame", "gyro", "accel"} & set(message)


def test_orientation_carries_its_frame(
    client: CyberwaveMQTTClient, paho: MagicMock
) -> None:
    client.publish_imu(TWIN, angular_velocity=GYRO, orientation=QUAT)
    _, message = _sent(paho)
    assert message["orientation"] == QUAT
    assert message["coordinate_frame"] == "environment"
    assert "linear_acceleration" not in message


def test_the_topic_carries_the_client_prefix(paho: MagicMock) -> None:
    client = CyberwaveMQTTClient(
        mqtt_broker="localhost",
        mqtt_port=1883,
        mqtt_username="user",
        api_key="api_key_secret",
        auto_connect=False,
        topic_prefix="dev",
    )
    client.connected = True
    client.publish_imu(TWIN, linear_acceleration=ACCEL)
    topic, _ = _sent(paho)
    assert topic == f"devcyberwave/twin/{TWIN}/imu"


@pytest.mark.skipif(
    not _FIXTURE.is_file(),
    reason="the imu contract lives in the monorepo's common/protobuf",
)
def test_every_field_sent_is_in_the_contract_fixture(
    client: CyberwaveMQTTClient, paho: MagicMock
) -> None:
    fixture = json.loads(_FIXTURE.read_text())
    payload = fixture["payload"]
    client.publish_imu(
        TWIN,
        angular_velocity=payload["angular_velocity"],
        linear_acceleration=payload["linear_acceleration"],
        orientation=payload["orientation"],
        sensor_id=payload["sensor_id"],
        timestamp=payload["timestamp"],
    )

    topic, message = _sent(paho)
    assert topic == fixture["topic"]
    # session_id is the client's, stamped on every publish.
    assert set(message) - {"session_id"} <= set(payload)
    for key in ("angular_velocity", "linear_acceleration", "orientation", "timestamp"):
        assert message[key] == payload[key]


def test_takes_sequences_as_drivers_hold_them(
    client: CyberwaveMQTTClient, paho: MagicMock
) -> None:
    client.publish_imu(
        TWIN,
        angular_velocity=(0.01, -0.02, 0.35),
        linear_acceleration=[0, 0, 9.81],
        orientation=[1, 0, 0, 0],
    )
    _, message = _sent(paho)
    assert message["angular_velocity"] == GYRO
    assert message["linear_acceleration"] == {"x": 0.0, "y": 0.0, "z": 9.81}
    assert message["orientation"] == {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}


def test_sends_only_the_contract_keys_as_floats(
    client: CyberwaveMQTTClient, paho: MagicMock
) -> None:
    client.publish_imu(TWIN, angular_velocity={"x": 1, "y": 2, "z": 3, "w": 9})
    _, message = _sent(paho)
    assert message["angular_velocity"] == {"x": 1.0, "y": 2.0, "z": 3.0}


def test_uses_the_hardware_timestamp(
    client: CyberwaveMQTTClient, paho: MagicMock
) -> None:
    client.publish_imu(TWIN, angular_velocity=GYRO, timestamp=1788433200)
    _, message = _sent(paho)
    assert message["timestamp"] == 1788433200.0


@pytest.mark.parametrize("twin_uuid", ["not-a-uuid", "", f"{TWIN}/extra"])
def test_refuses_a_twin_id_that_is_not_a_uuid(
    client: CyberwaveMQTTClient, paho: MagicMock, twin_uuid: str
) -> None:
    with pytest.raises(ValueError, match="twin_uuid must be a UUID"):
        client.publish_imu(twin_uuid, angular_velocity=GYRO)
    paho.publish.assert_not_called()


def test_refuses_a_sample_without_either_vector(
    client: CyberwaveMQTTClient, paho: MagicMock
) -> None:
    with pytest.raises(ValueError, match="angular_velocity, linear_acceleration"):
        client.publish_imu(TWIN, orientation=QUAT)
    paho.publish.assert_not_called()


def test_refuses_a_non_environment_frame(
    client: CyberwaveMQTTClient, paho: MagicMock
) -> None:
    with pytest.raises(ValueError, match="environment"):
        client.publish_imu(
            TWIN, angular_velocity=GYRO, orientation=QUAT, coordinate_frame="imu_init"
        )
    paho.publish.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"angular_velocity": {"x": 1.0, "y": 2.0}}, r"angular_velocity\.z"),
        ({"angular_velocity": [1.0, 2.0]}, "3 components"),
        ({"linear_acceleration": [0, 0, 9.81, 0]}, "3 components"),
        ({"linear_acceleration": {"x": float("nan"), "y": 0, "z": 0}}, r"\.x"),
        ({"linear_acceleration": {"x": float("inf"), "y": 0, "z": 0}}, r"\.x"),
        ({"angular_velocity": {"x": "1.0", "y": 0, "z": 0}}, r"angular_velocity\.x"),
        ({"angular_velocity": {"x": True, "y": 0, "z": 0}}, r"angular_velocity\.x"),
        ({"angular_velocity": "fast"}, "dict"),
        ({"angular_velocity": GYRO, "orientation": [0.0, 0.0, 1.0]}, "4 components"),
        (
            {"angular_velocity": GYRO, "orientation": {"x": 0, "y": 0, "z": 0}},
            r"orientation\.w",
        ),
        ({"angular_velocity": GYRO, "timestamp": float("nan")}, "timestamp"),
        ({"angular_velocity": GYRO, "timestamp": True}, "timestamp"),
        ({"angular_velocity": GYRO, "timestamp": "now"}, "timestamp"),
    ],
)
def test_refuses_a_sample_vector_would_drop(
    client: CyberwaveMQTTClient, paho: MagicMock, kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        client.publish_imu(TWIN, **kwargs)
    paho.publish.assert_not_called()


def test_source_type_can_be_overridden(
    client: CyberwaveMQTTClient, paho: MagicMock
) -> None:
    client.publish_imu(TWIN, angular_velocity=GYRO, source_type="sim")
    _, message = _sent(paho)
    assert message["source_type"] == "sim"


def test_refuses_an_unknown_source_type(
    client: CyberwaveMQTTClient, paho: MagicMock
) -> None:
    with pytest.raises(ValueError, match="source_type"):
        client.publish_imu(TWIN, angular_velocity=GYRO, source_type="robot")
    paho.publish.assert_not_called()


def test_sends_the_envelope_fields_only_when_given(
    client: CyberwaveMQTTClient, paho: MagicMock
) -> None:
    client.publish_imu(
        TWIN,
        angular_velocity=GYRO,
        source_subtype="mujoco",
        producer_instance="sim-7f3a",
        producer_sequence="77190",
        workload_uuid="c25a1938-16fa-47e7-b941-5792d375bd3d",
        simulation_time_s=12.5,
    )
    client.publish_imu(TWIN, angular_velocity=GYRO)

    (with_envelope, without) = [
        json.loads(c.args[1]) for c in paho.publish.call_args_list
    ]
    assert with_envelope["source_subtype"] == "mujoco"
    assert with_envelope["producer_instance"] == "sim-7f3a"
    assert with_envelope["producer_sequence"] == "77190"
    assert with_envelope["workload_uuid"] == "c25a1938-16fa-47e7-b941-5792d375bd3d"
    assert with_envelope["simulation_time_s"] == 12.5
    assert not {
        "source_subtype",
        "producer_instance",
        "producer_sequence",
        "workload_uuid",
        "simulation_time_s",
    } & set(without)


@pytest.mark.parametrize(
    "envelope",
    [
        {"producer_sequence": "009"},
        {"producer_sequence": 2**63},
        {"producer_sequence": True},
        {"producer_instance": 7},
        {"simulation_time_s": -1.0},
        {"simulation_time_s": float("nan")},
    ],
)
def test_refuses_an_envelope_vector_would_drop(
    client: CyberwaveMQTTClient, paho: MagicMock, envelope: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        client.publish_imu(TWIN, angular_velocity=GYRO, **envelope)
    paho.publish.assert_not_called()


def test_a_driver_topic_spec_resolves_to_the_same_topic() -> None:
    spec = TopicSpec(topic_slug=TWIN_IMU_TOPIC_SLUG, payload_schema_ref="ImuPayload")
    assert resolve_topic_path(spec, TWIN, prefix="dev") == (
        f"devcyberwave/twin/{TWIN}/imu"
    )


def test_the_wrapper_delegates() -> None:
    wrapper = WrapperMQTTClient.__new__(WrapperMQTTClient)
    wrapper._client = MagicMock()
    wrapper._client.publish_imu.return_value = True

    assert (
        wrapper.publish_imu(
            TWIN, angular_velocity=GYRO, sensor_id="imu", source_type="edge"
        )
        is True
    )
    wrapper._client.publish_imu.assert_called_once_with(
        TWIN,
        angular_velocity=GYRO,
        linear_acceleration=None,
        orientation=None,
        sensor_id="imu",
        coordinate_frame="environment",
        timestamp=None,
        source_type="edge",
        source_subtype=None,
        producer_instance=None,
        producer_sequence=None,
        workload_uuid=None,
        simulation_time_s=None,
    )
