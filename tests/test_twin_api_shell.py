"""PR1 shell tests: grouped handles, mock transport, describe()."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cyberwave.twin import LocomoteTwin, Twin, create_twin
from cyberwave.twin.transport import ResolvedOutbound


def _default_mqtt_metadata() -> dict:
    return {
        "mqtt": {
            "topics": {"cyberwave/twin/{twin_uuid}/command": {}},
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


def _make_locomote_twin(*, metadata: dict | None = None) -> LocomoteTwin:
    mqtt = MagicMock()
    mqtt.connected = True
    client = SimpleNamespace(
        mqtt=mqtt,
        assets=MagicMock(),
        config=SimpleNamespace(
            runtime_mode="live",
            source_type="tele",
            topic_prefix="",
        ),
        twins=SimpleNamespace(api=None),
    )
    twin_data = SimpleNamespace(
        uuid="twin-uuid",
        name="Go2",
        asset_uuid="asset-uuid",
        metadata=metadata or _default_mqtt_metadata(),
        capabilities={"can_locomote": True},
    )
    return LocomoteTwin(client, twin_data)


def test_package_import_unchanged() -> None:
    from cyberwave.twin import Twin as TwinCls, create_twin as factory

    assert TwinCls is not None
    assert factory is not None


def test_move_forward_delegates_to_locomotion_handle() -> None:
    twin = _make_locomote_twin()
    with patch.object(twin.locomotion, "move_forward") as mock_move:
        twin.move_forward(1.0)
    mock_move.assert_called_once_with(
        1.0, duration=1.0, rate_hz=20.0, source_type=None
    )


def test_locomotion_move_forward_calls_resolve_topic_and_payload() -> None:
    twin = _make_locomote_twin()
    with patch.object(twin, "_prepare_outbound_command"):
        with patch("cyberwave.twin.transport.time.sleep"):
            twin.locomotion.move_forward(2.0, duration=0.2, rate_hz=10)
    assert twin._outbound_log[0].command == "move_forward"
    assert twin._outbound_log[0].payload["data"]["linear_x"] == 2.0
    assert twin._outbound_log[-1].command == "stop"


def test_publish_resolved_calls_mqtt_and_logs() -> None:
    twin = _make_locomote_twin()
    with patch.object(twin, "_prepare_outbound_command"):
        with patch("cyberwave.twin.transport.time.sleep"):
            twin.locomotion.move_forward(1.0, duration=0.2, rate_hz=10)
    assert twin.client.mqtt.publish.call_count >= 1
    assert twin._outbound_log[0].command == "move_forward"
    assert twin._outbound_log[-1].command == "stop"


def test_describe_lists_grouped_and_flat_methods() -> None:
    twin = _make_locomote_twin()
    info = twin.describe()
    assert "handles" in info
    assert "commands" in info
    assert "locomotion" in info["handles"]
    assert "move_forward" in info["commands"]["mqtt"]["supported"]
    assert "move_forward" in info["commands"]["catalog_methods"]
    flat = info.get("flat_methods", [])
    assert "move_forward" in flat
    assert "commands" in flat


def test_commands_get_schema_reads_twin_metadata() -> None:
    twin = _make_locomote_twin()
    schema = twin.commands.get_schema()
    assert "move_forward" in schema["commands"]["supported"]


def test_resolve_topic_and_payload_returns_resolved_outbound() -> None:
    twin = _make_locomote_twin()
    resolved = twin._resolve_topic_and_payload(
        command="move_forward",
        data={"linear_x": 1.0},
    )
    assert isinstance(resolved, ResolvedOutbound)
    assert "twin-uuid" in resolved.topic
