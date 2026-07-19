"""Power/battery handle MQTT live-view read path."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from cyberwave.manifest.driver_config import TWIN_COMMAND_TOPIC_SLUG
from cyberwave.twin import LocomoteTwin


def _twin_with_power() -> LocomoteTwin:
    mqtt = MagicMock()
    mqtt.connected = True
    client = SimpleNamespace(
        mqtt=mqtt,
        config=SimpleNamespace(topic_prefix=""),
        twins=SimpleNamespace(api=None),
    )
    twin = LocomoteTwin(
        client,
        SimpleNamespace(
            uuid="pwr-twin-001",
            name="Power demo",
            capabilities={"can_locomote": True},
        ),
    )
    twin._driver_catalog_cache = {
        "mqtt": {
            "topics": {
                TWIN_COMMAND_TOPIC_SLUG: {},
                "cyberwave/twin/{twin_uuid}/battery/status": {"direction": "publish"},
            },
            "commands": {"supported": []},
        },
        "zenoh": {},
    }
    return twin


def test_power_get_returns_live_view_and_callback() -> None:
    twin = _twin_with_power()
    callbacks: dict[str, object] = {}

    def subscribe(topic: str, callback, **kwargs):  # type: ignore[no-untyped-def]
        callbacks[topic] = callback

    twin.client.mqtt.subscribe = subscribe  # type: ignore[attr-defined]

    handle = twin.power
    view = handle.get(timeout=0.0)
    assert view == {}
    seen: list[dict] = []
    view.on_update(lambda s: seen.append(s))
    topic = f"cyberwave/twin/{twin.uuid}/battery/status"
    callbacks[topic]({"percentage": 87})
    assert view["percentage"] == 87
    assert seen and seen[-1]["percentage"] == 87
    assert handle.get(timeout=0.0) is view
