"""SDK reads compiled driver/MQTT config from metadata only (no YAML)."""

from __future__ import annotations

from cyberwave.manifest.driver_config import (
    JOINT_UPDATE_TOPIC_SLUG,
    TWIN_COMMAND_TOPIC_SLUG,
    command_spec,
    command_specs,
    extract_driver_config_from_metadata,
    extract_mqtt_bundle_from_metadata,
    has_joint_update_topic,
    has_locomotion_commands,
    mqtt_topic_slugs,
    resolve_outbound_topic_slug,
    supported_mqtt_commands,
)

_MQTT_BUNDLE = {
    "schema_version": 1,
    "topics": {
        "cyberwave/twin/{twin_uuid}/command": {
            "direction": "both",
            "description": "command",
        },
    },
    "commands": {"supported": ["move_forward", "stop"]},
}


def test_extract_from_metadata_mqtt_key() -> None:
    assert extract_mqtt_bundle_from_metadata({"mqtt": _MQTT_BUNDLE}) == _MQTT_BUNDLE


def test_extract_from_metadata_driver_config_key() -> None:
    metadata = {"driver": {"config": _MQTT_BUNDLE}}
    assert extract_mqtt_bundle_from_metadata(metadata) == _MQTT_BUNDLE
    assert extract_driver_config_from_metadata(metadata) == _MQTT_BUNDLE


def test_extract_from_metadata_driver_config_nested_mqtt() -> None:
    metadata = {"driver": {"config": {"mqtt": _MQTT_BUNDLE, "extra": "x"}}}
    assert extract_mqtt_bundle_from_metadata(metadata) == _MQTT_BUNDLE


def test_extract_legacy_driver_config_mqtt() -> None:
    metadata = {"driver_config": {"mqtt": _MQTT_BUNDLE}}
    assert extract_mqtt_bundle_from_metadata(metadata) == _MQTT_BUNDLE


def test_helpers_on_bundle() -> None:
    assert supported_mqtt_commands(_MQTT_BUNDLE) == ["move_forward", "stop"]
    assert "cyberwave/twin/{twin_uuid}/command" in mqtt_topic_slugs(_MQTT_BUNDLE)
    assert has_locomotion_commands(_MQTT_BUNDLE) is True
    assert has_joint_update_topic(_MQTT_BUNDLE) is False


def test_command_specs_on_compiled_bundle() -> None:
    bundle = {
        "topics": _MQTT_BUNDLE["topics"],
        "commands": {
            "supported": ["move_forward", "takeoff"],
            "specs": {
                "move_forward": {"continuous": True, "rate_hz": 20},
                "takeoff": {},
            },
        },
    }
    assert command_specs(bundle)["move_forward"]["continuous"] is True
    assert command_spec(bundle, "takeoff") == {}
    assert command_spec(bundle, "missing") == {}


def test_resolve_outbound_topic_slug_from_catalog() -> None:
    assert (
        resolve_outbound_topic_slug(channel="twin_command", bundle=_MQTT_BUNDLE)
        == TWIN_COMMAND_TOPIC_SLUG
    )


def test_resolve_outbound_topic_slug_falls_back_when_no_topics() -> None:
    bundle = {"commands": {"supported": ["stop"]}}
    assert (
        resolve_outbound_topic_slug(channel="twin_command", bundle=bundle)
        == TWIN_COMMAND_TOPIC_SLUG
    )


def test_resolve_outbound_topic_slug_raises_when_slug_missing() -> None:
    bundle = {
        "topics": {"cyberwave/twin/{twin_uuid}/telemetry": {}},
        "commands": {"supported": ["stop"]},
    }
    try:
        resolve_outbound_topic_slug(channel="twin_command", bundle=bundle)
    except ValueError as exc:
        assert TWIN_COMMAND_TOPIC_SLUG in str(exc)
        assert "telemetry" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_resolve_outbound_topic_slug_joint_channel() -> None:
    joint_bundle = {
        "topics": {JOINT_UPDATE_TOPIC_SLUG: {}},
        "commands": {"supported": []},
    }
    assert (
        resolve_outbound_topic_slug(channel="joint_update", bundle=joint_bundle)
        == JOINT_UPDATE_TOPIC_SLUG
    )
