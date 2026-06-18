import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cyberwave.twin import LocomoteTwin


def _build_twin(*, runtime_mode: str = "live", source_type: str = "edge") -> tuple[LocomoteTwin, MagicMock]:
    mqtt_client = MagicMock()
    mqtt_client.connected = True
    assets = MagicMock()
    assets.get.return_value = SimpleNamespace(
        metadata={
            "mqtt": {
                "topics": {"cyberwave/twin/{twin_uuid}/command": {}},
                "commands": {
                    "supported": [
                        "move_forward",
                        "move_backward",
                        "turn_left",
                        "turn_right",
                        "stop",
                    ]
                },
            }
        }
    )
    client = SimpleNamespace(
        mqtt=mqtt_client,
        assets=assets,
        config=SimpleNamespace(
            source_type=source_type,
            runtime_mode=runtime_mode,
            topic_prefix="",
        ),
        twins=SimpleNamespace(api=None),
    )
    twin = LocomoteTwin(client, SimpleNamespace(uuid="twin-uuid", name="Twin", asset_uuid="a"))
    return twin, mqtt_client


def test_move_forward_normalizes_legacy_sim_source_type() -> None:
    twin, mqtt_client = _build_twin(source_type="tele")

    with patch.object(twin, "_prepare_outbound_command"):
        with patch("cyberwave.twin.transport.time.sleep"):
            twin.move_forward(1.0, duration=0.1, rate_hz=10, source_type="sim")

    assert mqtt_client.publish.call_count >= 1
    assert twin._outbound_log[0].payload["source_type"] == "sim_tele"
    assert twin._outbound_log[0].command == "move_forward"
    assert twin._outbound_log[-1].command == "stop"


def test_move_forward_uses_runtime_mode_control_source_type() -> None:
    twin, mqtt_client = _build_twin(runtime_mode="live", source_type="edge")

    with patch.object(twin, "_prepare_outbound_command"):
        with patch("cyberwave.twin.transport.time.sleep"):
            twin.move_forward(1.0, duration=0.1, rate_hz=10)

    assert mqtt_client.publish.call_count >= 1
    assert twin._outbound_log[0].payload["source_type"] == "tele"
    assert twin._outbound_log[-1].command == "stop"


def test_joint_set_defaults_sim_config_source_type_to_sim_tele() -> None:
    from cyberwave.twin.classes import JointTwin

    mqtt_client = MagicMock()
    assets = MagicMock()
    assets.get.return_value = SimpleNamespace(
        metadata={
            "mqtt": {
                "topics": {
                    "cyberwave/joint/{twin_uuid}/update": {},
                    "cyberwave/twin/{twin_uuid}/command": {},
                },
                "commands": {"supported": []},
            }
        }
    )
    client = SimpleNamespace(
        mqtt=mqtt_client,
        assets=assets,
        config=SimpleNamespace(runtime_mode="live", topic_prefix="", source_type="sim"),
        twins=SimpleNamespace(api=None),
    )
    twin = JointTwin(
        client, SimpleNamespace(uuid="arm-1", name="Arm", asset_uuid="asset-1")
    )
    with patch("cyberwave.twin.capabilities.joints.controllable_joint_names", return_value=["j1"]):
        with patch.object(twin, "_prepare_outbound_command"):
            twin.joints.set({"j1": 1.0})
    assert twin._outbound_log[-1].payload["source_type"] == "sim_tele"


@pytest.mark.parametrize(
    ("method_name", "args", "expected_command"),
    [
        ("move_backward", (1.0,), "move_backward"),
        ("turn_left", (1.5,), "turn_left"),
        ("turn_right", (1.5,), "turn_right"),
    ],
)
def test_locomotion_methods_use_simulation_control_source_type(
    method_name: str, args: tuple[float], expected_command: str
) -> None:
    twin, mqtt_client = _build_twin(runtime_mode="simulation", source_type="sim")

    with patch.object(twin, "_prepare_outbound_command"):
        with patch("cyberwave.twin.transport.time.sleep"):
            getattr(twin, method_name)(*args, duration=0.1, rate_hz=10)

    assert mqtt_client.publish.call_count >= 1
    assert twin._outbound_log[0].payload["source_type"] == "sim_tele"
    assert twin._outbound_log[0].command == expected_command
    assert twin._outbound_log[-1].command == "stop"


def test_move_forward_burst_then_stop() -> None:
    twin, mqtt_client = _build_twin(runtime_mode="live", source_type="edge")

    with patch.object(twin, "_prepare_outbound_command"):
        with patch("cyberwave.twin.transport.time.sleep"):
            twin.locomotion.move_forward(0.3, duration=0.2, rate_hz=10)

    commands = [entry.command for entry in twin._outbound_log]
    assert commands.count("move_forward") == 2
    assert commands[-1] == "stop"
    forward_payloads = [
        entry.payload["data"]
        for entry in twin._outbound_log
        if entry.command == "move_forward"
    ]
    assert forward_payloads[0]["linear_x"] == 0.3
    assert mqtt_client.publish.call_count == 3


def test_joint_set_uses_runtime_mode_control_source_type() -> None:
    from cyberwave.twin.classes import JointTwin

    mqtt_client = MagicMock()
    assets = MagicMock()
    assets.get.return_value = SimpleNamespace(
        metadata={
            "mqtt": {
                "topics": {
                    "cyberwave/joint/{twin_uuid}/update": {},
                    "cyberwave/twin/{twin_uuid}/command": {},
                },
                "commands": {"supported": []},
            }
        }
    )
    client = SimpleNamespace(
        mqtt=mqtt_client,
        assets=assets,
        config=SimpleNamespace(runtime_mode="live", topic_prefix="", source_type="edge"),
        twins=SimpleNamespace(api=None),
    )
    twin = JointTwin(
        client, SimpleNamespace(uuid="twin-uuid", name="Arm", asset_uuid="asset-1")
    )
    with patch("cyberwave.twin.capabilities.joints.controllable_joint_names", return_value=["joint_1"]):
        with patch.object(twin, "_prepare_outbound_command"):
            twin.joints.set("joint_1", 90.0)
    mqtt_client.update_joint_state.assert_not_called()
    assert twin._outbound_log[-1].payload["source_type"] == "tele"
