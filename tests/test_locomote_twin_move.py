"""Locomotion move() and flat delegation tests."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.twin_patch import patch_twin

from cyberwave.twin import LocomoteTwin


def _twin() -> LocomoteTwin:
    client = SimpleNamespace(
        mqtt=MagicMock(),
        assets=MagicMock(),
        config=SimpleNamespace(runtime_mode="live", topic_prefix="", source_type="tele"),
        twins=SimpleNamespace(api=None),
    )
    client.assets.get.return_value = SimpleNamespace(
        metadata={
            "mqtt": {
                "topics": {"cyberwave/twin/{twin_uuid}/command": {}},
                "commands": {"supported": ["move", "move_forward", "stop"]},
            }
        }
    )
    return LocomoteTwin(client, SimpleNamespace(uuid="go2", name="Go2", asset_uuid="a"))


def test_move_forward_publishes_linear_speed_burst() -> None:
    twin = _twin()
    with patch.object(twin, "_prepare_outbound_command"):
        with patch_twin("transport.time.sleep"):
            twin.move_forward(2.0, duration=0.2, rate_hz=10)
    assert twin._outbound_log[0].command == "move_forward"
    assert twin._outbound_log[0].payload["data"]["linear_x"] == 2.0
    assert twin._outbound_log[-1].command == "stop"


def test_move_publishes_command_payload() -> None:
    twin = _twin()
    with patch.object(twin, "_prepare_outbound_command"):
        twin.locomotion.move(distance=1.0)
    assert twin._outbound_log[-1].command == "move"


def test_move_calls_prepare_outbound_command() -> None:
    twin = _twin()
    with patch.object(twin, "_prepare_outbound_command") as gate:
        with patch_twin("transport.time.sleep"):
            twin.locomotion.move_forward(1.0, duration=0.2, rate_hz=10)
    assert gate.call_count >= 1


def test_move_forward_invokes_policy_gate_via_publish() -> None:
    twin = _twin()
    with patch.object(twin.policy, "ensure_attached") as ensure_mock:
        with patch_twin("transport.time.sleep"):
            twin.locomotion.move_forward(1.0, duration=0.2, rate_hz=10)
    assert ensure_mock.call_count >= 1
