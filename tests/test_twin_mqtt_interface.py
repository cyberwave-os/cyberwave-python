"""PR2 MQTT outbound: catalog schema, validation, topic resolution, and publish."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cyberwave.manifest.driver_config import JOINT_UPDATE_TOPIC_SLUG, TWIN_COMMAND_TOPIC_SLUG
from cyberwave.twin import FlyingTwin, LocomoteTwin
from cyberwave.twin.classes import JointTwin


def _make_locomote_twin(
    *,
    metadata: dict | None = None,
    topic_prefix: str = "",
) -> LocomoteTwin:
    mqtt = MagicMock()
    mqtt.connected = True
    default_metadata = {
        "mqtt": {
            "topics": {TWIN_COMMAND_TOPIC_SLUG: {}},
            "commands": {
                "supported": [
                    "move_forward",
                    "move_backward",
                    "turn_left",
                    "turn_right",
                    "stop",
                    "move",
                ]
            },
        }
    }
    client = SimpleNamespace(
        mqtt=mqtt,
        assets=MagicMock(),
        config=SimpleNamespace(
            runtime_mode="live",
            source_type="tele",
            topic_prefix=topic_prefix,
        ),
        twins=SimpleNamespace(api=None),
    )
    twin_data = SimpleNamespace(
        uuid="twin-uuid",
        name="Go2",
        asset_uuid="asset-uuid",
        metadata=metadata or default_metadata,
        capabilities={"can_locomote": True},
    )
    return LocomoteTwin(client, twin_data)


def test_get_schema_returns_twin_mqtt_bundle() -> None:
    twin = _make_locomote_twin()
    schema = twin.commands.get_schema()
    assert "move_forward" in schema["commands"]["supported"]
    assert TWIN_COMMAND_TOPIC_SLUG in schema["topics"]


def test_set_schema_persists_metadata_and_rebinds_commands() -> None:
    twin = _make_locomote_twin()
    updated_metadata = {
        "mqtt": {
            "topics": {TWIN_COMMAND_TOPIC_SLUG: {"direction": "both"}},
            "commands": {
                "supported": ["move_forward", "stop", "custom_ping"],
                "specs": {},
            },
        }
    }
    twin.client.twins = MagicMock()
    twin.client.twins.update.return_value = SimpleNamespace(
        uuid="twin-uuid",
        metadata=updated_metadata,
    )

    driver_yml = {
        "registry_ids": ["acme/go2"],
        "mqtt": {
            "schema_version": 1,
            "driver_family": "python",
            "twin": {
                "command": {
                    "direction": "both",
                    "payload_schema_ref": "TwinCommandPayload",
                    "description": "cmd",
                }
            },
            "commands": {"supported": ["custom_ping", "stop"]},
        },
    }
    schema = twin.commands.set_schema(driver_yml, merge=False)

    twin.client.twins.update.assert_called_once()
    call_metadata = twin.client.twins.update.call_args.kwargs["metadata"]
    assert "custom_ping" in call_metadata["mqtt"]["commands"]["supported"]
    assert "custom_ping" in schema["commands"]["supported"]
    assert hasattr(twin.commands, "custom_ping")
    assert "custom_ping" in twin.commands._bound_catalog_commands


def test_get_schema_cache_and_force_refresh() -> None:
    twin = _make_locomote_twin()
    first = twin.commands.get_schema()
    twin._data.metadata = {
        "mqtt": {
            "topics": {TWIN_COMMAND_TOPIC_SLUG: {}},
            "commands": {"supported": ["stop"]},
        }
    }
    second = twin.commands.get_schema()
    assert second["commands"]["supported"] == first["commands"]["supported"]

    refreshed = twin.commands.get_schema(force_refresh=True)
    assert refreshed["commands"]["supported"] == ["stop"]


def test_invalid_command_raises_with_allowed_list() -> None:
    twin = _make_locomote_twin()
    with patch.object(twin, "_prepare_outbound_command"):
        with pytest.raises(ValueError, match="move_forward"):
            twin._resolve_topic_and_payload(command="move_sideways", data={})


def test_resolve_topic_prefix_in_full_topic() -> None:
    twin = _make_locomote_twin(topic_prefix="dev/")
    resolved = twin._resolve_topic_and_payload(command="move_forward", data={"linear_x": 1.0})
    assert resolved.topic == "dev/cyberwave/twin/twin-uuid/command"


def test_publish_connects_mqtt_when_disconnected() -> None:
    twin = _make_locomote_twin()
    twin.client.mqtt.connected = False
    with patch.object(twin, "_prepare_outbound_command"):
        with patch("cyberwave.twin.transport.time.sleep"):
            twin.locomotion.move_forward(1.0, duration=0.2, rate_hz=10)
    assert twin.client.mqtt.connect.call_count >= 1
    assert twin.client.mqtt.publish.call_count >= 1


def test_publish_resolved_calls_mqtt_publish() -> None:
    twin = _make_locomote_twin()
    with patch.object(twin, "_prepare_outbound_command"):
        with patch("cyberwave.twin.transport.time.sleep"):
            twin.locomotion.move_forward(1.5, duration=0.2, rate_hz=10)
    first_topic, first_payload = twin.client.mqtt.publish.call_args_list[0][0]
    assert first_topic == "cyberwave/twin/twin-uuid/command"
    assert first_payload["command"] == "move_forward"
    assert first_payload["data"]["linear_x"] == 1.5
    assert twin._outbound_log[-1].command == "stop"


def test_joint_update_uses_joint_topic_slug() -> None:
    mqtt = MagicMock()
    mqtt.connected = True
    client = SimpleNamespace(
        mqtt=mqtt,
        assets=MagicMock(),
        config=SimpleNamespace(runtime_mode="live", topic_prefix="", source_type="tele"),
        twins=SimpleNamespace(api=None),
    )
    twin = JointTwin(
        client,
        SimpleNamespace(
            uuid="arm-1",
            name="Arm",
            asset_uuid="a",
            metadata={
                "mqtt": {
                    "topics": {
                        TWIN_COMMAND_TOPIC_SLUG: {},
                        JOINT_UPDATE_TOPIC_SLUG: {},
                    },
                    "commands": {"supported": []},
                }
            },
        ),
    )
    with patch("cyberwave.twin.capabilities.joints.controllable_joint_names", return_value=["j1"]):
        with patch.object(twin, "_prepare_outbound_command"):
            twin.joints.set({"j1": 90.0}, degrees=True)
    assert JOINT_UPDATE_TOPIC_SLUG.replace("{twin_uuid}", "arm-1") in twin._outbound_log[0].topic
    payload = twin._outbound_log[0].payload
    assert "command" not in payload
    assert payload["source_type"] == "tele"
    assert "j1" in payload
    mqtt.publish.assert_called_once()


def test_takeoff_publishes_when_in_catalog() -> None:
    mqtt = MagicMock()
    mqtt.connected = True
    client = SimpleNamespace(
        mqtt=mqtt,
        assets=MagicMock(),
        config=SimpleNamespace(runtime_mode="live", topic_prefix="", source_type="tele"),
        twins=SimpleNamespace(
            api=None,
            update=MagicMock(return_value=SimpleNamespace(uuid="drone-uuid", metadata={})),
        ),
    )
    twin = FlyingTwin(
        client,
        SimpleNamespace(
            uuid="drone-uuid",
            name="Drone",
            asset_uuid="a",
            metadata={
                "mqtt": {
                    "topics": {TWIN_COMMAND_TOPIC_SLUG: {}},
                    "commands": {"supported": ["takeoff", "land", "hover"]},
                }
            },
        ),
    )
    with patch.object(twin, "_prepare_outbound_command"):
        twin.takeoff(altitude=2.0)
    mqtt.publish.assert_called_once()
    assert twin._outbound_log[-1].command == "takeoff"
    assert twin._outbound_log[-1].payload["data"]["altitude"] == 2.0
