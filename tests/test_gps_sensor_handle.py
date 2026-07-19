"""GPS handle MQTT read path."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cyberwave.manifest.driver_config import TWIN_GPS_TOPIC_SLUG
from cyberwave.twin import LocomoteTwin
from cyberwave.twin.sensors.gps import normalize_gps_payload


def _twin_with_gps(*, catalog: dict | None = None) -> LocomoteTwin:
    mqtt = MagicMock()
    client = SimpleNamespace(
        mqtt=mqtt,
        config=SimpleNamespace(topic_prefix=""),
        twins=SimpleNamespace(api=None),
    )
    twin = LocomoteTwin(
        client,
        SimpleNamespace(
            uuid="gps-twin-001",
            name="GPS demo",
            capabilities={
                "can_locomote": True,
                "sensors": [{"id": "gps_main", "type": "gps"}],
            },
        ),
    )
    if catalog is not None:
        twin._mqtt_catalog_cache = catalog
    return twin


def test_normalize_gps_payload_shallow_copy() -> None:
    payload = {"latitude": 1.0, "longitude": 2.0}
    out = normalize_gps_payload(payload)
    assert out == payload
    assert out is not payload


def test_gps_get_fix_returns_latest_fix() -> None:
    twin = _twin_with_gps(
        catalog={
            "topics": {
                TWIN_GPS_TOPIC_SLUG: {
                    "direction": "publish",
                }
            }
        }
    )
    callbacks: dict[str, object] = {}

    def subscribe(topic: str, callback, **kwargs):  # type: ignore[no-untyped-def]
        callbacks[topic] = callback

    twin.client.mqtt.subscribe = subscribe  # type: ignore[attr-defined]
    twin.client.mqtt.connected = True

    handle = twin.gps[0]
    handle._ensure_gps_listeners()
    topic = f"cyberwave/twin/{twin.uuid}/gps"

    callbacks[topic](
        {
            "sensor_id": "gps_main",
            "latitude": 45.1,
            "longitude": 9.2,
            "altitude": 120.0,
        }
    )

    fix = handle.get_fix(timeout=1.0)
    assert fix["latitude"] == pytest.approx(45.1)
    assert fix["longitude"] == pytest.approx(9.2)


def test_gps_get_fix_filters_by_sensor_id() -> None:
    twin = _twin_with_gps()
    callbacks: dict[str, object] = {}

    def subscribe(topic: str, callback, **kwargs):  # type: ignore[no-untyped-def]
        callbacks[topic] = callback

    twin.client.mqtt.subscribe = subscribe  # type: ignore[attr-defined]
    twin.client.mqtt.connected = True

    handle = twin.gps[0]
    handle._ensure_gps_listeners()
    topic = f"cyberwave/twin/{twin.uuid}/gps"
    callbacks[topic]({"sensor_id": "other_gps", "latitude": 1.0, "longitude": 2.0})

    assert handle.get_fix(timeout=0.05) == {}  # wrong sensor_id dropped; empty view


def test_gps_get_fix_empty_view_without_message() -> None:
    twin = _twin_with_gps()
    twin.client.mqtt.subscribe = lambda *a, **k: None  # type: ignore[attr-defined]
    twin.client.mqtt.connected = True
    assert twin.gps.get_fix(timeout=0.05) == {}


def test_gps_get_fix_auto_refreshes_and_fires_callback() -> None:
    twin = _twin_with_gps()
    callbacks: dict[str, object] = {}

    def subscribe(topic: str, callback, **kwargs):  # type: ignore[no-untyped-def]
        callbacks[topic] = callback

    twin.client.mqtt.subscribe = subscribe  # type: ignore[attr-defined]
    twin.client.mqtt.connected = True

    handle = twin.gps
    view = handle.get_fix(timeout=0.0)
    seen: list[dict] = []
    view.on_update(lambda s: seen.append(s))
    topic = f"cyberwave/twin/{twin.uuid}/gps"
    callbacks[topic]({"latitude": 45.1, "longitude": 9.2, "altitude": 120.0})
    assert view["latitude"] == 45.1
    assert seen and seen[-1]["longitude"] == 9.2
    assert handle.get_fix(timeout=0.0) is view  # cached view
