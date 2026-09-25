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
