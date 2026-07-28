"""manifest.yaml top-level mqtt_max_hz parses into NodeManifest.mqtt_max_hz."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

from cyberwave.driver.ros2.manifest import _parse_manifest_root

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


def test_mqtt_max_hz_absent_is_none():
    m = _parse_manifest_root({"node_name": "n"})
    assert m.mqtt_max_hz is None


def test_mqtt_max_hz_parsed_as_float():
    m = _parse_manifest_root({"node_name": "n", "mqtt_max_hz": 20})
    assert m.mqtt_max_hz == 20.0


def test_mqtt_max_hz_non_positive_rejected():
    import pytest

    # 0 must NOT silently mean "unlimited" — the limiter treats max_hz <= 0 as
    # uncapped, so a manifest 0 would produce an outbound firehose.
    for bad in (0, -1):
        with pytest.raises(ValueError, match="mqtt_max_hz must be positive"):
            _parse_manifest_root({"node_name": "n", "mqtt_max_hz": bad})


from cyberwave.driver.ros2.manifest import NodeManifest


class _StubDriver:
    """Minimal stand-in to exercise the configure() mqtt_max_hz assignment."""
    def __init__(self, manifest):
        self._manifest = manifest
        self._mqtt_max_hz = None
        self._managed_launch = None


def test_configure_copies_manifest_mqtt_max_hz():
    from cyberwave.driver.ros2.base_ros2_driver import BaseROS2Driver
    d = _StubDriver(NodeManifest(node_name="n", mqtt_max_hz=12.0))
    BaseROS2Driver.configure(d)  # type: ignore[arg-type]
    assert d._mqtt_max_hz == 12.0
