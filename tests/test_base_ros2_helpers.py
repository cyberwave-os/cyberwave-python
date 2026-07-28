"""Pure-logic tests for BaseROS2Driver helpers (no rclpy node needed)."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

# rclpy is optional in CI; stub before importing BaseROS2Driver (see
# test_base_ros2_driver_managed_launch.py for the canonical stub set).
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

from cyberwave.driver.ros2.base_ros2_driver import BaseROS2Driver


class _NS:
    """Bind only the helpers under test onto a bare object."""
    namespaced = BaseROS2Driver.namespaced
    destroy_ros_publishers_for_topic = BaseROS2Driver.destroy_ros_publishers_for_topic

    def __init__(self, ns):
        self._ros_namespace = ns
        self.destroyed = []
        self._ros_out_publishers = {}

    def destroy_publisher(self, pub):
        self.destroyed.append(pub)


def test_namespaced_global_when_unset():
    assert _NS("").namespaced("enable_srv") == "/enable_srv"


def test_namespaced_under_twin_namespace():
    assert _NS("/CW_abc").namespaced("enable_srv") == "/CW_abc/enable_srv"


def test_destroy_publishers_for_topic_removes_only_matching():
    n = _NS("")
    n._ros_out_publishers = {
        ("/cmd", object): "pub_cmd",
        ("/other", object): "pub_other",
    }
    n.destroy_ros_publishers_for_topic("/cmd")
    assert n.destroyed == ["pub_cmd"]
    assert all(k[0] != "/cmd" for k in n._ros_out_publishers)
    assert len(n._ros_out_publishers) == 1
