"""TwinTelemetry facade and transport publish."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from cyberwave.telemetry.base import BaseTelemetry


class _FakeMqtt:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict[str, Any]]] = []
        self.connected = True

    def publish(self, topic: str, message: dict[str, Any]) -> None:
        self.published.append((topic, message))


def test_base_telemetry_publish_if_dirty() -> None:
    published: list[dict[str, Any]] = []

    tel = BaseTelemetry(
        publish_payload=published.append,
        snapshot_provider=lambda: {"lifecycle_state": "active"},
    )
    tel.update(operation_mode="no_op")
    assert tel.publish_if_dirty() is True
    assert published[0]["type"] == "driver_info"
    assert published[0]["lifecycle_state"] == "active"
    assert published[0]["operation_mode"] == "no_op"
    assert tel.publish_if_dirty() is False


def test_twin_telemetry_publish_uses_transport() -> None:
    mqtt = _FakeMqtt()
    client = MagicMock()
    client.mqtt = mqtt
    client.config = MagicMock(topic_prefix="")
    twin_data = MagicMock()
    twin_data.uuid = "abc-123"
    from cyberwave.twin.base import Twin

    twin = Twin(client, twin_data)
    twin.telemetry.publish({"type": "driver_info", "foo": 1})
    assert mqtt.published
    topic, payload = mqtt.published[-1]
    assert "abc-123" in topic
    assert "telemetry" in topic
    assert payload["foo"] == 1


def test_twin_telemetry_driver_info_not_implemented() -> None:
    mqtt = _FakeMqtt()
    client = MagicMock()
    client.mqtt = mqtt
    client.config = MagicMock(topic_prefix="")
    twin_data = MagicMock()
    twin_data.uuid = "abc-123"
    from cyberwave.twin.base import Twin

    twin = Twin(client, twin_data)
    with pytest.raises(NotImplementedError):
        twin.telemetry.driver_info(lifecycle_state="active")
