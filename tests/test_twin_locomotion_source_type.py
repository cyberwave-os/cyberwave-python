import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cyberwave.twin import LocomoteTwin


def _build_twin(*, runtime_mode: str = "live", source_type: str = "edge") -> tuple[LocomoteTwin, MagicMock]:
    mqtt_client = MagicMock()
    mqtt_client.connected = True
    client = SimpleNamespace(
        mqtt=mqtt_client,
        config=SimpleNamespace(
            source_type=source_type,
            runtime_mode=runtime_mode,
            topic_prefix="",
        ),
    )
    twin = LocomoteTwin(client, SimpleNamespace(uuid="twin-uuid", name="Twin"))
    return twin, mqtt_client


def test_move_forward_normalizes_legacy_sim_source_type() -> None:
    twin, mqtt_client = _build_twin(source_type="tele")

    twin.move_forward(1.0, source_type="sim")

    mqtt_client.publish.assert_called_once()
    topic, payload = mqtt_client.publish.call_args.args
    assert topic == "cyberwave/twin/twin-uuid/command"
    assert payload["source_type"] == "sim_tele"
    assert payload["command"] == "move_forward"


def test_move_forward_uses_runtime_mode_control_source_type() -> None:
    twin, mqtt_client = _build_twin(runtime_mode="live", source_type="edge")

    twin.move_forward(1.0)

    mqtt_client.publish.assert_called_once()
    _, payload = mqtt_client.publish.call_args.args
    assert payload["source_type"] == "tele"


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

    getattr(twin, method_name)(*args)

    mqtt_client.publish.assert_called_once()
    topic, payload = mqtt_client.publish.call_args.args
    assert topic == "cyberwave/twin/twin-uuid/command"
    assert payload["source_type"] == "sim_tele"
    assert payload["command"] == expected_command


def test_joint_set_uses_runtime_mode_control_source_type() -> None:
    twin, mqtt_client = _build_twin(runtime_mode="live", source_type="edge")

    twin.joints.set("joint_1", 90.0)

    mqtt_client.update_joint_state.assert_called_once()
    call = mqtt_client.update_joint_state.call_args
    assert call.args[:2] == ("twin-uuid", "joint_1")
    assert math.isclose(call.kwargs["position"], math.pi / 2)
    assert call.kwargs["source_type"] == "tele"
