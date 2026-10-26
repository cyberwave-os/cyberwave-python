"""``BaseEdgeNode.publish_health`` stamps the node's configured ``source_type``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from cyberwave.edge.base import BaseEdgeNode
from cyberwave.edge.config import EdgeNodeConfig


class _Node(BaseEdgeNode):
    def __init__(self, config: EdgeNodeConfig, health: Dict[str, Any]) -> None:
        super().__init__(config)
        self._health = health

    async def _setup(self) -> None: ...

    async def _subscribe_to_commands(self) -> None: ...

    async def _cleanup(self) -> None: ...

    def _build_health_status(self) -> Dict[str, Any]:
        return dict(self._health)


def _published_payload(source_type: str, health: Dict[str, Any]) -> Dict[str, Any]:
    node = _Node(EdgeNodeConfig(edge_uuid="edge-1", source_type=source_type), health)
    node.client = SimpleNamespace(mqtt=MagicMock(topic_prefix=""))

    node.publish_health("twin-a", node._build_health_status())

    topic, payload = node.client.mqtt.publish.call_args.args
    assert topic == "cyberwave/twin/twin-a/edge_health"
    return payload


@pytest.mark.parametrize("source_type", ["edge", "sim"])
def test_heartbeat_carries_the_configured_source_type(source_type: str) -> None:
    payload = _published_payload(source_type, {"status": "ok"})

    assert payload["source_type"] == source_type
    assert payload["status"] == "ok"


def test_configured_source_type_wins_over_health_data() -> None:
    """Same rule as ``EdgeHealthCheck.build_payload``: config beats the body."""
    payload = _published_payload("sim", {"source_type": "edge"})

    assert payload["source_type"] == "sim"


def test_an_unknown_source_type_is_published_as_edge() -> None:
    """Same fallback as BaseDriver: an unvalidated env value is never stamped."""
    payload = _published_payload("simm", {"status": "ok"})

    assert payload["source_type"] == "edge"


def test_heartbeat_uses_the_standard_edge_health_keys() -> None:
    """The keys ``EdgeHealthCheck.build_payload`` and the frontend hooks use."""
    payload = _published_payload("edge", {"status": "ok"})

    assert payload["type"] == "edge_health"
    assert payload["edge_id"] == "edge-1"
    assert payload["twin_uuid"] == "twin-a"
    assert isinstance(payload["uptime_seconds"], float)
    assert payload["uptime_seconds"] >= 0
    assert "edge_uuid" not in payload
    assert "uptime" not in payload


@pytest.mark.parametrize(
    ("configured", "source_type"),
    [
        ("edge", "edge"),
        ("sim", "sim"),
        ("sim_tele", "sim_tele"),
        # The backend moves a physical action only on ``edge``; the hardware
        # roles would otherwise have every status update dropped.
        ("edge_leader", "edge"),
        ("edge_follower", "edge"),
        # A typo falls back to ``edge``, as on the heartbeat.
        ("simm", "edge"),
    ],
)
def test_navigate_status_carries_the_configured_source_type(
    configured: str, source_type: str
) -> None:
    node = _Node(EdgeNodeConfig(edge_uuid="edge-1", source_type=configured), {})
    node.client = SimpleNamespace(mqtt=MagicMock(topic_prefix=""))

    node.publish_nav_status("twin-a", "action-1", "running", progress=50.0)

    topic, payload = node.client.mqtt.publish.call_args.args
    assert topic == "cyberwave/twin/twin-a/navigate/status"
    assert payload["source_type"] == source_type
    assert payload["action_id"] == "action-1"
    assert payload["status"] == "running"


@pytest.mark.parametrize("configured", ["tele", "edit"])
def test_navigate_status_from_a_command_origin_carries_no_source_type(
    configured: str,
) -> None:
    """The backend drops ``tele``/``edit`` on a physical action, not a missing source."""
    node = _Node(EdgeNodeConfig(edge_uuid="edge-1", source_type=configured), {})
    node.client = SimpleNamespace(mqtt=MagicMock(topic_prefix=""))

    node.publish_nav_status("twin-a", "action-1", "completed")

    _topic, payload = node.client.mqtt.publish.call_args.args
    assert "source_type" not in payload
    assert payload["action_id"] == "action-1"
    assert payload["status"] == "completed"
