"""``twin.driver`` catalog getters and ``set_schema``."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from cyberwave.manifest.driver_config import TWIN_COMMAND_TOPIC_SLUG
from cyberwave.twin import LocomoteTwin


def _make_twin(*, metadata: dict | None = None) -> LocomoteTwin:
    mqtt = MagicMock()
    mqtt.connected = True
    default_metadata = {
        "mqtt": {
            "topics": {TWIN_COMMAND_TOPIC_SLUG: {}},
            "commands": {"supported": ["move_forward", "stop"]},
        }
    }
    client = SimpleNamespace(
        mqtt=mqtt,
        assets=MagicMock(),
        config=SimpleNamespace(
            runtime_mode="live", source_type="tele", topic_prefix=""
        ),
        twins=SimpleNamespace(api=None),
    )
    twin_data = SimpleNamespace(
        uuid="twin-uuid",
        name="Bot",
        asset_uuid="asset-uuid",
        metadata=metadata or default_metadata,
        capabilities={"can_locomote": True},
    )
    return LocomoteTwin(client, twin_data)


def test_driver_getters_from_metadata() -> None:
    twin = _make_twin()
    assert twin.driver.get_supported_commands() == ["move_forward", "stop"]
    assert TWIN_COMMAND_TOPIC_SLUG in twin.driver.get_supported_topics()
    assert twin.driver.get_supported_transports() == ["mqtt"]
    assert twin.driver.get_supported_channels() == []

    schemas = twin.driver.get_schemas()
    assert "mqtt" in schemas
    assert "zenoh" in schemas
    assert "move_forward" in schemas["mqtt"]["commands"]["supported"]


def test_driver_set_schema_persists_mqtt_and_rebinds_commands() -> None:
    twin = _make_twin()
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
    twin.client.twins.set_driver_schema.return_value = SimpleNamespace(
        uuid="twin-uuid",
        metadata=updated_metadata,
    )

    driver_yml = {
        "registry_id": "acme/go2",
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
    schemas = twin.driver.set_schema(driver_yml, merge=False)

    twin.client.twins.set_driver_schema.assert_called_once()
    call_kwargs = twin.client.twins.set_driver_schema.call_args.kwargs
    assert call_kwargs["merge"] is False
    assert (
        "custom_ping" in call_kwargs["driver_config"]["mqtt"]["commands"]["supported"]
    )
    call_metadata = updated_metadata
    assert "custom_ping" in call_metadata["mqtt"]["commands"]["supported"]
    assert "custom_ping" in schemas["mqtt"]["commands"]["supported"]
    assert hasattr(twin.commands, "custom_ping")


def test_reviewed_schema_write_forwards_revision_and_retains_state_on_conflict():
    from datetime import datetime, timezone

    import pytest

    from cyberwave.exceptions import CyberwaveError

    twin = _make_twin()
    twin._data.updated_at = datetime(2026, 9, 10, tzinfo=timezone.utc)
    before = twin._data
    schemas = twin.driver.get_schemas()
    twin.client.twins = MagicMock()
    twin.client.twins.set_driver_schema.side_effect = RuntimeError(
        "409 configuration changed"
    )
    root = {"mqtt": {"twin": {"command": {"description": "Commands"}}}}
    with pytest.raises(CyberwaveError, match="409 configuration changed"):
        twin.driver.set_schema(root, expected_revision=twin.driver.get_revision())
    twin.client.twins.set_driver_schema.assert_called_once_with(
        twin.uuid,
        driver_config=root,
        merge=True,
        expected_revision="2026-09-10T00:00:00+00:00",
    )
    assert twin._data is before
    assert twin.driver.get_schemas() == schemas
    assert twin.driver.get_supported_commands() == ["move_forward", "stop"]


def test_driver_revision_legacy_absence_and_refreshed_value():
    twin = _make_twin()
    assert twin.driver.get_revision() is None
    twin.client.twins = MagicMock()
    twin.client.twins.get_raw.return_value = SimpleNamespace(
        uuid=twin.uuid,
        metadata=twin._data.metadata,
        updated_at="2026-09-11T00:00:00Z",
    )
    twin.refresh()
    assert twin.driver.get_revision() == "2026-09-11T00:00:00Z"


def test_driver_revision_resource_serialization_preserves_legacy_wire_shape():
    from cyberwave.resources import TwinManager

    api = MagicMock()
    manager = TwinManager(api)
    root = {"mqtt": {"commands": {"supported": ["stop"]}}}
    manager.set_driver_schema("twin-id", driver_config=root, merge=False)
    assert api.api_client.param_serialize.call_args.kwargs["body"] == {
        "driver_config": root,
        "merge": False,
    }
    manager.set_driver_schema(
        "twin-id", driver_config=root, expected_revision="reviewed-token"
    )
    assert api.api_client.param_serialize.call_args.kwargs["body"] == {
        "driver_config": root,
        "merge": True,
        "expected_revision": "reviewed-token",
    }


def test_driver_uses_backend_projection_and_refresh_rebinds_command_methods():
    twin = _make_twin()
    commands = twin.commands
    assert hasattr(commands, "move_forward")
    resolved = {
        "topics": {TWIN_COMMAND_TOPIC_SLUG: {}},
        "commands": {
            "supported": ["custom_ping"],
            "specs": {
                "custom_ping": {
                    "args": [{"name": "count", "type": "integer", "default": 0}]
                }
            },
        },
    }
    twin.client.twins = MagicMock()
    twin.client.twins.get_raw.return_value = SimpleNamespace(
        uuid=twin.uuid,
        metadata={},
        mqtt_command_schema=resolved,
    )
    twin.refresh()
    assert twin.commands is commands
    assert hasattr(commands, "custom_ping")
    assert not hasattr(commands, "move_forward")
    assert twin.driver.get_supported_commands() == ["custom_ping"]
    assert twin.driver.get_supported_transports() == ["mqtt"]
    assert twin.driver.get_command_specs()["custom_ping"]["args"][0]["default"] == 0
    view = twin.driver.get_mqtt_schema()
    view["commands"]["supported"].clear()
    assert twin.driver.get_supported_commands() == ["custom_ping"]
    twin.client.mqtt.publish.assert_not_called()


def test_authoritative_empty_projection_does_not_revive_stale_metadata():
    twin = _make_twin()
    twin._data.mqtt_command_schema = {"topics": {}, "commands": {"supported": []}}
    assert twin.driver.get_supported_commands() == []
    assert twin.driver.get_supported_transports() == []
    assert twin.driver.get_mqtt_schema()["topics"] == {}
