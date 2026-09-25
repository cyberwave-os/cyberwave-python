"""Tests for update_twin_gps rate limiting (2 Hz per twin)."""

import time
from unittest.mock import MagicMock, patch

import pytest

from cyberwave.mqtt import CyberwaveMQTTClient

TWIN_UUID = "test-twin-uuid"
GPS_MIN_UPDATE_INTERVAL = 0.5  # must match mqtt/__init__.py _gps_min_update_interval


@pytest.fixture
def mqtt_client():
    with patch("cyberwave.mqtt.mqtt.Client"):
        client = CyberwaveMQTTClient(
            mqtt_broker="localhost",
            mqtt_port=1883,
            mqtt_username="user",
            api_key="test-api-key",
            auto_connect=False,
            source_type="edge",
        )
    client.publish = MagicMock()
    return client


def test_first_gps_update_is_published(mqtt_client):
    mqtt_client.update_twin_gps(TWIN_UUID, 37.0, -122.0, fix_type="3d")
    mqtt_client.publish.assert_called_once()


def test_immediate_second_gps_update_is_rate_limited(mqtt_client):
    mqtt_client.update_twin_gps(TWIN_UUID, 37.0, -122.0, fix_type="3d")
    mqtt_client.update_twin_gps(TWIN_UUID, 38.0, -123.0, fix_type="3d")
    assert mqtt_client.publish.call_count == 1


def test_gps_update_after_interval_is_published(mqtt_client):
    rate_key = f"twin:{TWIN_UUID}:gps"
    mqtt_client._last_update_times[rate_key] = (
        time.monotonic() - GPS_MIN_UPDATE_INTERVAL - 0.001
    )

    mqtt_client.update_twin_gps(TWIN_UUID, 37.0, -122.0, fix_type="3d")
    mqtt_client.publish.assert_called_once()


def test_fix_type_none_is_dropped_without_consuming_rate_limit(mqtt_client):
    mqtt_client.update_twin_gps(TWIN_UUID, 0.0, 0.0, fix_type="none")
    mqtt_client.publish.assert_not_called()

    mqtt_client.update_twin_gps(TWIN_UUID, 37.0, -122.0, fix_type="3d")
    mqtt_client.publish.assert_called_once()


def test_gps_rate_limits_are_independent_per_twin(mqtt_client):
    mqtt_client.update_twin_gps("twin-a", 1.0, 2.0, fix_type="3d")
    mqtt_client.update_twin_gps("twin-b", 3.0, 4.0, fix_type="3d")
    assert mqtt_client.publish.call_count == 2
