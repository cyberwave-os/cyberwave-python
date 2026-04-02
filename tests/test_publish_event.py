"""Tests for Cyberwave.publish_event() and config.twin_uuid."""

from unittest.mock import MagicMock, patch

import pytest

from cyberwave.config import CyberwaveConfig


@pytest.fixture
def cw_client():
    """Create a Cyberwave client with mocked REST + MQTT."""
    with (
        patch("cyberwave.client.ApiClient"),
        patch("cyberwave.client.DefaultApi"),
        patch("cyberwave.client.Configuration"),
    ):
        from cyberwave.client import Cyberwave

        client = Cyberwave(api_key="test-key", base_url="http://localhost:8000")

        mock_mqtt = MagicMock()
        mock_mqtt.topic_prefix = ""
        client._mqtt_client = mock_mqtt

        return client


def test_publish_event_correct_topic(cw_client):
    cw_client.publish_event("twin-uuid", "person_detected", {"count": 3})

    cw_client._mqtt_client.publish.assert_called_once()
    call_args = cw_client._mqtt_client.publish.call_args
    topic = call_args[0][0]
    assert topic == "cyberwave/twin/twin-uuid/event"


def test_publish_event_payload_shape(cw_client):
    cw_client.publish_event("twin-uuid", "person_detected", {"count": 3})

    call_args = cw_client._mqtt_client.publish.call_args
    payload = call_args[0][1]
    assert "event_type" in payload
    assert "source" in payload
    assert "data" in payload
    assert "timestamp" in payload
    assert payload["event_type"] == "person_detected"
    assert payload["data"] == {"count": 3}
    assert isinstance(payload["timestamp"], float)


def test_publish_event_default_source(cw_client):
    cw_client.publish_event("twin-uuid", "evt", {})

    payload = cw_client._mqtt_client.publish.call_args[0][1]
    assert payload["source"] == "edge_node"


def test_publish_event_custom_source(cw_client):
    cw_client.publish_event("twin-uuid", "evt", {}, source="sensor")

    payload = cw_client._mqtt_client.publish.call_args[0][1]
    assert payload["source"] == "sensor"


def test_publish_event_with_topic_prefix(cw_client):
    cw_client._mqtt_client.topic_prefix = "staging/"
    cw_client.publish_event("twin-uuid", "evt", {})

    topic = cw_client._mqtt_client.publish.call_args[0][0]
    assert topic == "staging/cyberwave/twin/twin-uuid/event"


# ── config.twin_uuid ──────────────────────────────────────────


def test_config_twin_uuid_from_env(monkeypatch):
    """twin_uuid should be populated from CYBERWAVE_TWIN_UUID."""
    monkeypatch.setenv("CYBERWAVE_TWIN_UUID", "abc-123")
    cfg = CyberwaveConfig()
    assert cfg.twin_uuid == "abc-123"


def test_config_twin_uuid_default_none(monkeypatch):
    """twin_uuid defaults to None when env var is absent."""
    monkeypatch.delenv("CYBERWAVE_TWIN_UUID", raising=False)
    cfg = CyberwaveConfig()
    assert cfg.twin_uuid is None


def test_config_twin_uuid_explicit_overrides_env(monkeypatch):
    """Explicit twin_uuid kwarg takes precedence over env var."""
    monkeypatch.setenv("CYBERWAVE_TWIN_UUID", "from-env")
    cfg = CyberwaveConfig(twin_uuid="explicit")
    assert cfg.twin_uuid == "explicit"
