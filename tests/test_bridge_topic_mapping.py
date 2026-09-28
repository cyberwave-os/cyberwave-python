"""Tests for Zenoh ↔ MQTT topic mapping."""

import pytest

from cyberwave.zenoh_mqtt.topic_mapping import (
    build_inbound_subscriptions,
    build_outbound_subscriptions,
    mqtt_to_zenoh,
    zenoh_to_mqtt,
)

TWIN = "12345678-1234-1234-1234-123456789abc"


class TestZenohToMqtt:
    def test_basic_channel(self):
        result = zenoh_to_mqtt(f"cw/{TWIN}/data/model_output")
        assert result is not None
        assert result.mqtt_topic == f"cyberwave/twin/{TWIN}/model_output"
        assert result.twin_uuid == TWIN
        assert result.channel == "model_output"

    def test_channel_with_sensor(self):
        result = zenoh_to_mqtt(f"cw/{TWIN}/data/model_output/default")
        assert result is not None
        assert result.mqtt_topic == f"cyberwave/twin/{TWIN}/model_output"
        assert result.channel == "model_output"

    def test_event_channel(self):
        result = zenoh_to_mqtt(f"cw/{TWIN}/data/event")
        assert result is not None
        assert result.mqtt_topic == f"cyberwave/twin/{TWIN}/event"

    def test_model_health_channel(self):
        result = zenoh_to_mqtt(f"cw/{TWIN}/data/model_health")
        assert result is not None
        assert result.mqtt_topic == f"cyberwave/twin/{TWIN}/model_health"

    def test_with_mqtt_prefix(self):
        result = zenoh_to_mqtt(f"cw/{TWIN}/data/event", mqtt_prefix="staging")
        assert result is not None
        assert result.mqtt_topic == f"stagingcyberwave/twin/{TWIN}/event"

    def test_invalid_uuid_returns_none(self):
        assert zenoh_to_mqtt("cw/not-a-uuid/data/event") is None

    def test_missing_data_segment_returns_none(self):
        assert zenoh_to_mqtt(f"cw/{TWIN}/stream/event") is None

    def test_too_short_returns_none(self):
        assert zenoh_to_mqtt(f"cw/{TWIN}") is None

    def test_empty_string_returns_none(self):
        assert zenoh_to_mqtt("") is None


class TestMqttToZenoh:
    def test_basic_topic(self):
        result = mqtt_to_zenoh(f"cyberwave/twin/{TWIN}/commands/sync_workflows")
        assert result is not None
        assert result.zenoh_key == f"cw/{TWIN}/data/commands_sync_workflows"
        assert result.twin_uuid == TWIN
        assert result.channel == "commands_sync_workflows"

    def test_single_segment_suffix(self):
        result = mqtt_to_zenoh(f"cyberwave/twin/{TWIN}/event")
        assert result is not None
        assert result.zenoh_key == f"cw/{TWIN}/data/event"
        assert result.channel == "event"

    def test_with_mqtt_prefix(self):
        result = mqtt_to_zenoh(
            f"stagingcyberwave/twin/{TWIN}/event",
            mqtt_prefix="staging",
        )
        assert result is not None
        assert result.zenoh_key == f"cw/{TWIN}/data/event"

    def test_custom_zenoh_prefix(self):
        result = mqtt_to_zenoh(
            f"cyberwave/twin/{TWIN}/event",
            zenoh_prefix="myprefix",
        )
        assert result is not None
        assert result.zenoh_key == f"myprefix/{TWIN}/data/event"

    def test_invalid_uuid_returns_none(self):
        assert mqtt_to_zenoh("cyberwave/twin/bad-uuid/event") is None

    def test_wrong_prefix_returns_none(self):
        assert mqtt_to_zenoh(f"other/twin/{TWIN}/event") is None

    def test_no_suffix_returns_none(self):
        assert mqtt_to_zenoh(f"cyberwave/twin/{TWIN}/") is None

    def test_empty_returns_none(self):
        assert mqtt_to_zenoh("") is None


class TestBuildOutboundSubscriptions:
    def test_single_twin_multiple_channels(self):
        keys = build_outbound_subscriptions(
            [TWIN], ["model_output", "event"]
        )
        assert f"cw/{TWIN}/data/model_output" in keys
        assert f"cw/{TWIN}/data/model_output/*" in keys
        assert f"cw/{TWIN}/data/event" in keys
        assert f"cw/{TWIN}/data/event/*" in keys
        assert len(keys) == 4

    def test_custom_prefix(self):
        keys = build_outbound_subscriptions(
            [TWIN], ["event"], zenoh_prefix="myapp"
        )
        assert f"myapp/{TWIN}/data/event" in keys

    def test_empty_twins(self):
        assert build_outbound_subscriptions([], ["event"]) == []

    def test_empty_channels(self):
        assert build_outbound_subscriptions([TWIN], []) == []


class TestBuildInboundSubscriptions:
    def test_single_twin_single_topic(self):
        topics = build_inbound_subscriptions(
            [TWIN], ["commands/sync_workflows"]
        )
        assert topics == [f"cyberwave/twin/{TWIN}/commands/sync_workflows"]

    def test_with_prefix(self):
        topics = build_inbound_subscriptions(
            [TWIN], ["commands/sync_workflows"], mqtt_prefix="dev"
        )
        assert topics == [f"devcyberwave/twin/{TWIN}/commands/sync_workflows"]

    def test_empty_twins(self):
        assert build_inbound_subscriptions([], ["commands/sync"]) == []

    def test_empty_topics(self):
        assert build_inbound_subscriptions([TWIN], []) == []
