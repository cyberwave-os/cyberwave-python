"""IMU handle MQTT read path."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cyberwave.manifest.driver_config import TWIN_IMU_TOPIC_SLUG
from cyberwave.twin import LocomoteTwin
from cyberwave.twin.sensors.imu import normalize_imu_payload


def _twin_with_imu(*, catalog: dict | None = None) -> LocomoteTwin:
    mqtt = MagicMock()
    client = SimpleNamespace(
        mqtt=mqtt,
        config=SimpleNamespace(topic_prefix=""),
        twins=SimpleNamespace(api=None),
    )
    twin = LocomoteTwin(
        client,
        SimpleNamespace(
            uuid="imu-twin-001",
            name="IMU demo",
            capabilities={
                "can_locomote": True,
                "sensors": [{"id": "d455_imu", "type": "imu"}],
            },
        ),
    )
    if catalog is not None:
        twin._mqtt_catalog_cache = catalog
    return twin


def test_normalize_imu_payload_aliases() -> None:
    out = normalize_imu_payload(
        {
            "angular_velocity": {"x": 1.0, "y": 0.0, "z": 0.0},
            "linear_acceleration": {"x": 0.0, "y": 0.0, "z": 9.81},
        }
    )
    assert out["gyro"]["x"] == pytest.approx(1.0)
    assert out["accel"]["z"] == pytest.approx(9.81)


def test_imu_get_returns_latest_sample() -> None:
    twin = _twin_with_imu(
        catalog={
            "topics": {
                TWIN_IMU_TOPIC_SLUG: {
                    "direction": "publish",
                    "payload_schema_ref": "ImuPayload",
                }
            }
        }
    )
    callbacks: dict[str, object] = {}

    def subscribe(topic: str, callback, **kwargs):  # type: ignore[no-untyped-def]
        callbacks[topic] = callback

    twin.client.mqtt.subscribe = subscribe  # type: ignore[attr-defined]
    twin.client.mqtt.connected = True

    handle = twin.imu[0]
    handle._ensure_imu_listeners()
    topic = f"cyberwave/twin/{twin.uuid}/imu"

    callbacks[topic](
        {
            "sensor_id": "d455_imu",
            "gyro": {"x": 0.1, "y": 0.2, "z": 0.3},
            "accel": {"x": 0.0, "y": 0.0, "z": 9.81},
        }
    )

    sample = handle.get(timeout=1.0)
    assert sample["gyro"]["z"] == pytest.approx(0.3)
    assert sample["accel"]["z"] == pytest.approx(9.81)
    assert "angular_velocity" not in sample
    assert "linear_acceleration" not in sample


def test_imu_get_filters_by_sensor_id() -> None:
    twin = _twin_with_imu()
    callbacks: dict[str, object] = {}

    def subscribe(topic: str, callback, **kwargs):  # type: ignore[no-untyped-def]
        callbacks[topic] = callback

    twin.client.mqtt.subscribe = subscribe  # type: ignore[attr-defined]
    twin.client.mqtt.connected = True

    handle = twin.imu[0]
    handle._ensure_imu_listeners()
    topic = f"cyberwave/twin/{twin.uuid}/imu"
    callbacks[topic](
        {
            "sensor_id": "other_imu",
            "gyro": {"x": 9.0, "y": 0.0, "z": 0.0},
            "accel": {"x": 0.0, "y": 0.0, "z": 9.81},
        }
    )

    view = handle.get(timeout=0.05)
    assert view == {}  # wrong sensor_id dropped; empty live view, no raise


def test_imu_get_auto_refreshes_and_fires_callback() -> None:
    twin = _twin_with_imu()
    callbacks: dict[str, object] = {}

    def subscribe(topic: str, callback, **kwargs):  # type: ignore[no-untyped-def]
        callbacks[topic] = callback

    twin.client.mqtt.subscribe = subscribe  # type: ignore[attr-defined]
    twin.client.mqtt.connected = True

    handle = twin.imu[0]
    view = handle.get(timeout=0.0)
    seen: list[dict] = []
    view.on_update(lambda s: seen.append(s))
    topic = f"cyberwave/twin/{twin.uuid}/imu"
    callbacks[topic](
        {"gyro": {"x": 0.0, "y": 0.0, "z": 0.3}, "accel": {"x": 0.0, "y": 0.0, "z": 9.81}}
    )
    assert view["gyro"]["z"] == 0.3  # same object refreshed
    assert seen and seen[-1]["accel"]["z"] == 9.81
    assert handle.get(timeout=0.0) is view  # cached view
