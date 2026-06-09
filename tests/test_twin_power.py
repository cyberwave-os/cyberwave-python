from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cyberwave.twin import LocomoteTwin


def test_power_check_publishes_battery_check() -> None:
    client = SimpleNamespace(
        mqtt=MagicMock(),
        assets=MagicMock(),
        config=SimpleNamespace(runtime_mode="live", topic_prefix=""),
        twins=SimpleNamespace(api=None),
    )
    twin = LocomoteTwin(
        client,
        SimpleNamespace(
            uuid="dji",
            name="Drone",
            asset_uuid="a",
            capabilities={"can_locomote": True},
        ),
    )
    with patch.object(twin, "_prepare_outbound_command"):
        twin.power.check()
    assert twin._outbound_log[-1].command == "battery_check"
