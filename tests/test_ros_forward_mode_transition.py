"""ROS feedback shared by operation modes must stay subscribed."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from cyberwave.driver.interface.args import DriverOperationMode
from cyberwave.driver.interface.registry_mixin import InterfaceRegistryMixin


def _driver(current_entries, desired_entries):
    driver = MagicMock()
    driver._operation_mode = DriverOperationMode.NO_OP
    driver._ros_forward_handles = [
        SimpleNamespace(entry=entry) for entry in current_entries
    ]
    driver._interface.ros_forward_publishers_for_mode.return_value = desired_entries
    driver._is_interface_wired.return_value = False
    driver.on_exit_operation = AsyncMock()
    driver.on_enter_teleop_local = AsyncMock()
    return driver


def test_shared_ros_forward_survives_mode_change() -> None:
    shared_entry = object()
    driver = _driver([shared_entry], [shared_entry])

    with patch("cyberwave.driver.ros2.ros_publishers.wire_ros_publishers") as wire:
        asyncio.run(
            InterfaceRegistryMixin._set_operation_mode(
                driver, DriverOperationMode.TELEOP_LOCAL
            )
        )

    wire.assert_not_called()
    driver.on_enter_teleop_local.assert_awaited_once()
    assert driver._ros_forward_handles[0].entry is shared_entry


def test_ros_forward_rewires_when_mode_changes_publishers() -> None:
    driver = _driver([object()], [object()])

    with patch("cyberwave.driver.ros2.ros_publishers.wire_ros_publishers") as wire:
        asyncio.run(
            InterfaceRegistryMixin._set_operation_mode(
                driver, DriverOperationMode.TELEOP_LOCAL
            )
        )

    wire.assert_called_once_with(driver)
