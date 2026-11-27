"""CameraStreamMixin composed with BaseROS2Driver: the robot bridge shape (rclpy stubbed)."""

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


from cyberwave.driver import CameraSourceConfig, CameraStreamMixin  # noqa: E402
from cyberwave.driver.ros2.base_ros2_driver import BaseROS2Driver  # noqa: E402
from cyberwave.driver.sensors.camera import CameraStreamMixin as _Mixin  # noqa: E402


class RobotBridge(CameraStreamMixin, BaseROS2Driver):
    REGISTRY_ID = "unitree/go2"
    CAMERA_SOURCE = CameraSourceConfig.push(camera_name="front_camera")


def test_the_mixin_owns_the_ros2_drivers_monitoring_phase() -> None:
    # BaseROS2Driver.run_async gathers self.on_start_monitoring() while ACTIVE.
    assert RobotBridge.on_start_monitoring is _Mixin.on_start_monitoring


def test_the_robot_keeps_its_heartbeat_and_lifecycle() -> None:
    assert RobotBridge.EDGE_HEALTH_ENABLED is True
    assert RobotBridge.VIDEO_COMMANDS_DRIVE_LIFECYCLE is False


def test_the_declared_source_is_used() -> None:
    bridge = RobotBridge.__new__(RobotBridge)
    bridge._twin = None

    config = bridge.camera_source_config()

    assert config.is_pre_encoded and config.gst_source == "appsrc"
    assert config.camera_name == "front_camera"


def test_push_before_the_stream_opens_is_dropped_not_raised() -> None:
    bridge = RobotBridge.__new__(RobotBridge)

    assert bridge.push_h264(b"\x00\x00\x00\x01\x65") is False
