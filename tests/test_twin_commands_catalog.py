"""Catalog-driven methods on ``twin.commands`` (command factory)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cyberwave.twin import FlyingTwin, LocomoteTwin
from cyberwave.twin.command_factory import resolve_command_delegate


def _dji_metadata() -> dict:
    return {
        "mqtt": {
            "topics": {"cyberwave/twin/{twin_uuid}/command": {}},
            "commands": {
                "supported": [
                    "takeoff",
                    "land",
                    "gimbal_rotate",
                    "set_gimbal_pitch",
                    "gimbal_rotate_speed",
                    "emergency_stop",
                ],
            },
        },
    }


def _go2_metadata() -> dict:
    locomotion_specs = {
        name: {"continuous": True}
        for name in ("move_forward", "move_backward", "turn_left", "turn_right")
    }
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
                    "camera_up",
                ],
                "specs": {
                    **locomotion_specs,
                    "stop": {},
                    "camera_up": {},
                },
            },
        },
    }


def _so101_metadata() -> dict:
    return {
        "mqtt": {
            "topics": {
                "cyberwave/twin/{twin_uuid}/command": {},
                "cyberwave/joint/{twin_uuid}/update": {},
            },
            "commands": {
                "supported": [
                    "remoteoperate",
                    "teleoperate",
                    "recalibrate",
                    "calibrate",
                    "stop",
                ],
            },
        },
    }


def _make_twin(
    twin_cls: type,
    *,
    metadata: dict,
    capabilities: dict | None = None,
) -> LocomoteTwin | FlyingTwin:
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
        name="Robot",
        asset_uuid="asset-uuid",
        metadata=metadata,
        capabilities=capabilities or {},
    )
    return twin_cls(client, twin_data)


def test_commands_handle_binds_catalog_methods_at_init() -> None:
    twin = _make_twin(FlyingTwin, metadata=_dji_metadata(), capabilities={"can_fly": True})
    handle = twin.commands
    assert "gimbal_rotate" in handle._bound_catalog_commands
    assert callable(handle.gimbal_rotate)
    assert callable(handle.takeoff)
    assert "gimbal_rotate" in dir(handle)


def test_resolve_command_delegate_prefers_locomotion() -> None:
    twin = _make_twin(
        LocomoteTwin,
        metadata=_go2_metadata(),
        capabilities={"can_locomote": True},
    )
    assert resolve_command_delegate(twin, "move_forward") == (
        "locomotion",
        twin.locomotion.move_forward,
    )
    assert resolve_command_delegate(twin, "camera_up") is None


def test_catalog_locomotion_command_burst_delegates() -> None:
    twin = _make_twin(
        LocomoteTwin,
        metadata={
            "mqtt": {
                "topics": {"cyberwave/twin/{twin_uuid}/command": {}},
                "commands": {
                    "supported": ["turn_left", "stop"],
                    "specs": {
                        "turn_left": {"continuous": True},
                        "stop": {},
                    },
                },
            },
        },
        capabilities={"can_locomote": True},
    )
    assert twin.commands._command_routing["turn_left"]["via"] == "burst"
    assert twin.commands._command_routing["turn_left"]["continuous"] is True
    assert twin.commands._command_routing["stop"]["via"] == "locomotion.stop"

    with patch.object(twin, "_prepare_outbound_command"):
        with patch("cyberwave.twin.transport.time.sleep"):
            twin.commands.turn_left(angular_z=0.4, duration=0.2, rate_hz=10)

    commands = [entry.command for entry in twin._outbound_log]
    assert commands.count("turn_left") == 2
    assert commands[-1] == "stop"
    turn_payloads = [
        entry.payload["data"]
        for entry in twin._outbound_log
        if entry.command == "turn_left"
    ]
    assert turn_payloads[0]["angular_z"] == 0.4


def test_catalog_go2_move_forward_delegates_burst() -> None:
    twin = _make_twin(
        LocomoteTwin,
        metadata=_go2_metadata(),
        capabilities={"can_locomote": True},
    )
    with patch.object(twin, "_prepare_outbound_command"):
        with patch("cyberwave.twin.transport.time.sleep"):
            twin.commands.move_forward(linear_x=0.5, duration=0.2, rate_hz=10)

    forward = [
        entry for entry in twin._outbound_log if entry.command == "move_forward"
    ]
    assert len(forward) == 2
    assert forward[0].payload["data"]["linear_x"] == 0.5
    assert twin._outbound_log[-1].command == "stop"


def test_catalog_camera_up_stays_raw_mqtt_publish() -> None:
    twin = _make_twin(
        LocomoteTwin,
        metadata=_go2_metadata(),
        capabilities={"can_locomote": True},
    )
    assert twin.commands._command_routing["camera_up"]["via"] == "mqtt_publish"

    with patch.object(twin, "_prepare_outbound_command"):
        twin.commands.camera_up(tilt=0.1)

    assert len(twin._outbound_log) == 1
    entry = twin._outbound_log[0]
    assert entry.command == "camera_up"
    assert entry.payload["data"]["tilt"] == 0.1


def test_catalog_flight_command_delegates_to_flight_handle() -> None:
    twin = _make_twin(
        FlyingTwin,
        metadata=_dji_metadata(),
        capabilities={"can_fly": True},
    )
    assert twin.commands._command_routing["takeoff"]["via"] == "flight.takeoff"

    with patch.object(twin, "_prepare_outbound_command"):
        twin.commands.takeoff(altitude=2.0)

    entry = twin._outbound_log[0]
    assert entry.command == "takeoff"
    assert entry.payload["data"]["altitude"] == 2.0


def test_catalog_command_merges_data_and_kwargs() -> None:
    twin = _make_twin(
        FlyingTwin,
        metadata=_dji_metadata(),
        capabilities={"can_fly": True},
    )
    with patch.object(twin, "_prepare_outbound_command"):
        twin.commands.gimbal_rotate(pitch=-45.0, duration=1.5)
    entry = twin._outbound_log[0]
    assert entry.command == "gimbal_rotate"
    assert entry.payload["data"]["pitch"] == -45.0
    assert entry.payload["data"]["duration"] == 1.5


def test_catalog_command_rejects_unsupported_command() -> None:
    twin = _make_twin(
        LocomoteTwin,
        metadata={
            "mqtt": {
                "topics": {"cyberwave/twin/{twin_uuid}/command": {}},
                "commands": {"supported": ["stop"]},
            },
        },
        capabilities={"can_locomote": True},
    )
    with pytest.raises(AttributeError):
        twin.commands.turn_left  # noqa: B018 — not bound


def test_describe_lists_catalog_command_methods() -> None:
    twin = _make_twin(
        LocomoteTwin,
        metadata=_so101_metadata(),
        capabilities={"can_locomote": True},
    )
    info = twin.describe()
    assert "commands" in info
    assert info["commands"] is info["handles"]["commands"]
    methods = info["commands"]["methods"]
    assert "get_schema" in methods
    assert "teleoperate" in methods
    assert "teleoperate" in info["commands"]["catalog_methods"]
    assert info["commands"]["access"] == "twin.commands"
    assert "teleoperate" in info["commands"]["mqtt"]["supported"]
    assert info["commands"]["mqtt"]["has_joint_update_topic"] is True
    assert "publish" in info["commands"]
    assert "commands" in info["flat_methods"]


def test_catalog_dji_continuous_ascend_bursts() -> None:
    twin = _make_twin(
        FlyingTwin,
        metadata={
            "mqtt": {
                "topics": {"cyberwave/twin/{twin_uuid}/command": {}},
                "commands": {
                    "supported": ["ascend", "takeoff"],
                    "specs": {
                        "ascend": {"continuous": True, "rate_hz": 20},
                        "takeoff": {},
                    },
                },
            },
        },
        capabilities={"can_fly": True, "can_locomote": True},
    )
    assert twin.commands._command_routing["ascend"]["via"] == "burst"

    with patch.object(twin, "_prepare_outbound_command"):
        with patch("cyberwave.twin.transport.time.sleep"):
            twin.commands.ascend(linear_z=1.5, duration=0.2, rate_hz=10)

    commands = [entry.command for entry in twin._outbound_log]
    assert commands.count("ascend") == 2
    assert commands[-1] == "stop"


def test_describe_includes_command_routing_for_locomote() -> None:
    twin = _make_twin(
        LocomoteTwin,
        metadata=_go2_metadata(),
        capabilities={"can_locomote": True},
    )
    routing = twin.describe()["commands"]["command_routing"]
    assert routing["move_forward"]["via"] == "burst"
    assert routing["move_forward"]["continuous"] is True
    assert routing["camera_up"]["via"] == "mqtt_publish"
    assert "specs" in twin.describe()["commands"]["mqtt"]
