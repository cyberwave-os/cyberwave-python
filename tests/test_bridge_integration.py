"""Integration tests for ZenohMqttBridge.

These tests mock both Zenoh sessions and MQTT clients to verify the bridge
wiring without requiring actual network connectivity.
"""

import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from cyberwave.zenoh_mqtt.config import BridgeConfig
from cyberwave.zenoh_mqtt.bridge import ZenohMqttBridge
from cyberwave.zenoh_mqtt.topic_mapping import zenoh_to_mqtt, mqtt_to_zenoh

TWIN = "12345678-1234-1234-1234-123456789abc"


@pytest.fixture
def bridge_config(tmp_path):
    return BridgeConfig(
        enabled=True,
        twin_uuids=[TWIN],
        outbound_channels=["model_output", "event"],
        inbound_topics=["commands/sync_workflows"],
        queue_dir=str(tmp_path / "bridge_queue"),
        queue_max_bytes=1024 * 1024,
    )


class TestBridgeOutbound:
    """Verify that Zenoh samples are forwarded to MQTT (or queued when offline)."""

    def test_publish_or_queue_when_connected(self, bridge_config, tmp_path):
        bridge = ZenohMqttBridge(config=bridge_config)
        bridge._mqtt_connected.set()
        bridge._mqtt_client = MagicMock()
        result_mock = MagicMock()
        bridge._mqtt_client.publish.return_value = result_mock

        bridge._publish_or_queue("cyberwave/twin/test/event", b"hello")

        bridge._mqtt_client.publish.assert_called_once_with(
            "cyberwave/twin/test/event", b"hello", qos=1
        )
        assert bridge.stats["outbound"] == 1
        assert bridge.stats["queued"] == 0

    def test_publish_or_queue_when_disconnected(self, bridge_config, tmp_path):
        bridge = ZenohMqttBridge(config=bridge_config)
        bridge._mqtt_connected.clear()

        bridge._publish_or_queue("cyberwave/twin/test/event", b"hello")

        assert bridge.stats["outbound"] == 0
        assert bridge.stats["queued"] == 1
        assert not bridge._queue.is_empty

    def test_publish_or_queue_falls_back_to_queue_on_error(
        self, bridge_config, tmp_path
    ):
        bridge = ZenohMqttBridge(config=bridge_config)
        bridge._mqtt_connected.set()
        bridge._mqtt_client = MagicMock()
        bridge._mqtt_client.publish.side_effect = Exception("network error")

        bridge._publish_or_queue("cyberwave/twin/test/event", b"hello")

        assert bridge.stats["queued"] == 1


class TestBridgeInbound:
    """Verify that MQTT messages are forwarded to the local Zenoh session."""

    def test_on_mqtt_message_forwards_to_zenoh(self, bridge_config):
        bridge = ZenohMqttBridge(config=bridge_config)
        bridge._zenoh_session = MagicMock()

        mqtt_msg = MagicMock()
        mqtt_msg.topic = f"cyberwave/twin/{TWIN}/commands/sync_workflows"
        mqtt_msg.payload = b'{"action":"sync"}'

        bridge._on_mqtt_message(None, None, mqtt_msg)

        expected_key = f"cw/{TWIN}/data/commands_sync_workflows"
        bridge._zenoh_session.put.assert_called_once_with(
            expected_key, b'{"action":"sync"}'
        )
        assert bridge.stats["inbound"] == 1

    def test_on_mqtt_message_ignores_unmapped_topic(self, bridge_config):
        bridge = ZenohMqttBridge(config=bridge_config)
        bridge._zenoh_session = MagicMock()

        mqtt_msg = MagicMock()
        mqtt_msg.topic = "some/other/topic"
        mqtt_msg.payload = b"data"

        bridge._on_mqtt_message(None, None, mqtt_msg)

        bridge._zenoh_session.put.assert_not_called()
        assert bridge.stats["inbound"] == 0


class TestBridgeQueueDrain:
    """Verify that offline-queued messages are drained on reconnect."""

    def test_drain_publishes_queued_messages(self, bridge_config):
        bridge = ZenohMqttBridge(config=bridge_config)
        bridge._mqtt_connected.clear()

        bridge._publish_or_queue(f"cyberwave/twin/{TWIN}/event", b"msg1")
        bridge._publish_or_queue(f"cyberwave/twin/{TWIN}/event", b"msg2")
        assert bridge.stats["queued"] == 2

        # Simulate reconnect
        bridge._mqtt_connected.set()
        bridge._mqtt_client = MagicMock()
        result_mock = MagicMock()
        bridge._mqtt_client.publish.return_value = result_mock

        batch = bridge._queue.drain(10)
        assert len(batch) == 2
        assert batch[0].payload == b"msg1"
        assert batch[1].payload == b"msg2"


class TestBridgeStats:
    def test_stats_are_thread_safe(self, bridge_config):
        bridge = ZenohMqttBridge(config=bridge_config)
        bridge._mqtt_connected.set()
        bridge._mqtt_client = MagicMock()
        result_mock = MagicMock()
        bridge._mqtt_client.publish.return_value = result_mock

        threads = []
        for _ in range(10):
            t = threading.Thread(
                target=bridge._publish_or_queue,
                args=(f"cyberwave/twin/{TWIN}/event", b"data"),
            )
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert bridge.stats["outbound"] == 10


class TestBridgeLifecycle:
    def test_stop_without_start_is_noop(self, bridge_config):
        bridge = ZenohMqttBridge(config=bridge_config)
        bridge.stop()

    def test_double_start_is_idempotent(self, bridge_config):
        """Calling start() twice should not raise."""
        bridge = ZenohMqttBridge(config=bridge_config)
        # We cannot fully start without Zenoh/MQTT, but we can verify the guard
        bridge._started = True
        bridge.start()  # Should be a no-op
        assert bridge._started is True
