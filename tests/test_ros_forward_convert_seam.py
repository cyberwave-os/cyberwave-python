"""convert_joints_to_payload seam: default equals the full-joint converter."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# rclpy stub so importing base_ros2_driver does not require a ROS install.
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
    sys.modules["lifecycle_msgs.msg"].Transition = MagicMock
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

from cyberwave.driver.ros2.base_ros2_driver import BaseROS2Driver  # noqa: E402
from cyberwave.driver.ros2.message_payload import (  # noqa: E402
    ros_joint_state_to_transport_payload,
)


def test_default_seam_equals_full_joint_converter() -> None:
    driver = BaseROS2Driver.__new__(BaseROS2Driver)
    msg = SimpleNamespace(
        name=["j1", "j2"], position=[0.1, 0.2], velocity=[], effort=[], header=None
    )
    # msg.header is None, so ros_joint_state_to_transport_payload falls back to
    # time.time() for the "timestamp" field. Freeze it so the two independent
    # calls below (one via the seam, one via the direct converter) are
    # comparable rather than flaking on sub-millisecond timestamp drift.
    with patch("cyberwave.driver.ros2.message_payload.time.time", return_value=1234.0):
        assert driver.convert_joints_to_payload(msg) == ros_joint_state_to_transport_payload(
            msg
        )
