from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if "rclpy" not in sys.modules:
    _rclpy = types.ModuleType("rclpy")
    _rclpy.ok = lambda: True
    _rclpy.init = lambda *a, **k: None
    sys.modules["rclpy"] = _rclpy
    sys.modules["rclpy.executors"] = types.ModuleType("rclpy.executors")
    sys.modules["rclpy.executors"].MultiThreadedExecutor = MagicMock
    sys.modules["rclpy.qos"] = types.ModuleType("rclpy.qos")
    sys.modules["rclpy.qos"].QoSProfile = MagicMock
    sys.modules["rclpy.lifecycle"] = types.ModuleType("rclpy.lifecycle")
    sys.modules["rclpy.lifecycle"].LifecycleNode = object
    sys.modules["rclpy.lifecycle"].State = MagicMock
    sys.modules["rclpy.lifecycle"].TransitionCallbackReturn = MagicMock
    sys.modules["rclpy.parameter"] = types.ModuleType("rclpy.parameter")
    sys.modules["rclpy.parameter"].Parameter = MagicMock
    sys.modules["lifecycle_msgs"] = types.ModuleType("lifecycle_msgs")
    sys.modules["lifecycle_msgs.msg"] = types.ModuleType("lifecycle_msgs.msg")
    _transition = MagicMock()
    _transition.TRANSITION_CONFIGURE = 1
    _transition.TRANSITION_ACTIVATE = 3
    _transition.TRANSITION_DEACTIVATE = 4
    sys.modules["lifecycle_msgs.msg"].Transition = _transition
    sys.modules["lifecycle_msgs.srv"] = types.ModuleType("lifecycle_msgs.srv")
    sys.modules["lifecycle_msgs.srv"].ChangeState = MagicMock
    sys.modules["rcl_interfaces"] = types.ModuleType("rcl_interfaces")
    sys.modules["rcl_interfaces.msg"] = types.ModuleType("rcl_interfaces.msg")
    sys.modules["rcl_interfaces.msg"].ParameterDescriptor = MagicMock
    sys.modules["rcl_interfaces.msg"].ParameterType = MagicMock
    sys.modules["rcl_interfaces.msg"].SetParametersResult = MagicMock
    sys.modules["std_msgs"] = types.ModuleType("std_msgs")
    sys.modules["std_msgs.msg"] = types.ModuleType("std_msgs.msg")
    sys.modules["std_msgs.msg"].String = MagicMock

from lifecycle_msgs.msg import Transition

from cyberwave.driver.ros2.base_ros2_driver import BaseROS2Driver


def _minimal_ros_driver() -> BaseROS2Driver:
    class _D(BaseROS2Driver):
        REGISTRY_ID = "test/ros"

    driver = _D.__new__(_D)
    driver._ros_request_transition_sync = MagicMock()
    return driver


@pytest.mark.asyncio
async def test_on_enter_no_op_requests_deactivate() -> None:
    driver = _minimal_ros_driver()
    with patch.object(
        driver, "_ros_request_transition", new_callable=AsyncMock
    ) as req:
        await BaseROS2Driver.on_enter_no_op(driver)
        req.assert_awaited_once_with(Transition.TRANSITION_DEACTIVATE)


@pytest.mark.asyncio
async def test_on_enter_teleop_local_requests_activate() -> None:
    driver = _minimal_ros_driver()
    with patch.object(
        driver, "_ros_request_transition", new_callable=AsyncMock
    ) as req:
        await BaseROS2Driver.on_enter_teleop_local(driver)
        req.assert_awaited_once_with(Transition.TRANSITION_ACTIVATE)


def test_ros_request_transition_sync_skips_deactivate_when_not_active() -> None:
    driver = _minimal_ros_driver()
    driver._trigger_transition_sync = MagicMock()
    with patch(
        "cyberwave.driver.ros2.base_ros2_driver._ros_lifecycle_state_label",
        return_value="inactive",
    ):
        driver._ros_request_transition_sync(Transition.TRANSITION_DEACTIVATE)
    driver._trigger_transition_sync.assert_not_called()


def test_ros_request_transition_sync_skips_activate_when_already_active() -> None:
    driver = _minimal_ros_driver()
    driver._trigger_transition_sync = MagicMock()
    with patch(
        "cyberwave.driver.ros2.base_ros2_driver._ros_lifecycle_state_label",
        return_value="active",
    ):
        driver._ros_request_transition_sync(Transition.TRANSITION_ACTIVATE)
    driver._trigger_transition_sync.assert_not_called()


@pytest.mark.asyncio
async def test_run_async_connect_log_tolerates_unbound_twin() -> None:
    """Regression: the connect-phase log must use _resolve_twin_uuid(), not
    twin_uuid, so startup does not crash before the twin is fetched."""
    import asyncio

    from cyberwave.driver.base import DriverLifecycleState

    driver = _minimal_ros_driver()
    driver._twin = None
    driver._cw = None
    driver._cloud_connected = False
    driver._lifecycle_state = DriverLifecycleState.UNCONFIGURED
    driver._shutdown = asyncio.Event()
    driver._alert_manager = MagicMock()
    driver._emit_driver_info = MagicMock()
    driver._emit_lifecycle_alerts = MagicMock()
    driver._end_driver_telemetry_session = MagicMock()
    driver._run_shutdown = AsyncMock()
    driver._unwire_interface_from_registry = AsyncMock()
    driver._disconnect_cloud_client = MagicMock()

    class _Sentinel(Exception):
        pass

    driver._connect_cloud_async = AsyncMock(side_effect=_Sentinel)

    with patch(
        "cyberwave.driver.ros2.base_ros2_driver.unwire_ros_publishers"
    ):
        # With the bug, twin_uuid raises AttributeError before reaching
        # _connect_cloud_async; with the fix, the sentinel propagates instead.
        with pytest.raises(_Sentinel):
            await BaseROS2Driver.run_async(driver)

    driver._connect_cloud_async.assert_awaited_once()
    # finally must mirror BaseDriver.run_async and release the SDK client.
    driver._disconnect_cloud_client.assert_called_once()
