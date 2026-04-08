"""Tests for BridgeConfig environment-driven settings."""

import os
from unittest.mock import patch

import pytest

from cyberwave.zenoh_mqtt.config import BridgeConfig


class TestBridgeConfig:
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            for key in list(os.environ):
                if key.startswith("CYBERWAVE_BRIDGE") or key.startswith("CYBERWAVE_MQTT"):
                    os.environ.pop(key, None)
            cfg = BridgeConfig()
            assert cfg.enabled is False
            assert cfg.outbound_channels == ["model_output", "event", "model_health"]
            assert cfg.inbound_topics == ["commands/sync_workflows"]
            assert cfg.zenoh_key_prefix == "cw"
            assert cfg.mqtt_topic_prefix == ""
            assert cfg.queue_max_bytes == 50 * 1024 * 1024
            assert cfg.drain_batch_size == 64
            assert cfg.mqtt_qos == 1

    def test_enabled_from_env(self):
        with patch.dict(os.environ, {"CYBERWAVE_BRIDGE_ENABLED": "true"}):
            cfg = BridgeConfig()
            assert cfg.enabled is True

    def test_enabled_from_env_false(self):
        with patch.dict(os.environ, {"CYBERWAVE_BRIDGE_ENABLED": "0"}):
            cfg = BridgeConfig()
            assert cfg.enabled is False

    def test_twin_uuids_from_env(self):
        uuids = "abc-123, def-456"
        with patch.dict(os.environ, {"CYBERWAVE_BRIDGE_TWIN_UUIDS": uuids}):
            cfg = BridgeConfig()
            assert cfg.twin_uuids == ["abc-123", "def-456"]

    def test_outbound_channels_from_env(self):
        with patch.dict(
            os.environ,
            {"CYBERWAVE_BRIDGE_OUTBOUND_CHANNELS": "frames,depth,custom"},
        ):
            cfg = BridgeConfig()
            assert cfg.outbound_channels == ["frames", "depth", "custom"]

    def test_inbound_topics_from_env(self):
        with patch.dict(
            os.environ,
            {"CYBERWAVE_BRIDGE_INBOUND_TOPICS": "commands/sync,commands/deploy"},
        ):
            cfg = BridgeConfig()
            assert cfg.inbound_topics == ["commands/sync", "commands/deploy"]

    def test_mqtt_prefix_from_env(self):
        with patch.dict(os.environ, {"CYBERWAVE_MQTT_TOPIC_PREFIX": "staging"}):
            cfg = BridgeConfig()
            assert cfg.mqtt_topic_prefix == "staging"

    def test_queue_dir_from_env(self):
        with patch.dict(
            os.environ,
            {"CYBERWAVE_BRIDGE_QUEUE_DIR": "/var/cyberwave/queue"},
        ):
            cfg = BridgeConfig()
            assert cfg.queue_dir == "/var/cyberwave/queue"

    def test_queue_max_bytes_from_env(self):
        with patch.dict(
            os.environ,
            {"CYBERWAVE_BRIDGE_QUEUE_MAX_BYTES": "10485760"},
        ):
            cfg = BridgeConfig()
            assert cfg.queue_max_bytes == 10_485_760

    def test_explicit_values_override_env(self):
        with patch.dict(os.environ, {"CYBERWAVE_BRIDGE_ENABLED": "false"}):
            cfg = BridgeConfig(enabled=True, twin_uuids=["test-uuid"])
            assert cfg.enabled is True
            assert cfg.twin_uuids == ["test-uuid"]

    def test_drain_batch_size_from_env(self):
        with patch.dict(os.environ, {"CYBERWAVE_BRIDGE_DRAIN_BATCH_SIZE": "128"}):
            cfg = BridgeConfig()
            assert cfg.drain_batch_size == 128

    def test_mqtt_qos_from_env(self):
        with patch.dict(os.environ, {"CYBERWAVE_BRIDGE_MQTT_QOS": "0"}):
            cfg = BridgeConfig()
            assert cfg.mqtt_qos == 0
