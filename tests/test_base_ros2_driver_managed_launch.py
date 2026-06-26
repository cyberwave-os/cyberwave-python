from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

# rclpy is optional in CI; stub before importing BaseROS2Driver.
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
from cyberwave.driver.ros2.manifest import ManifestManagedLaunch, ManifestReadiness


def test_base_connect_to_device_starts_managed_launch_once() -> None:
    instance = MagicMock()
    instance.is_running = False

    class _D(BaseROS2Driver):
        REGISTRY_ID = "test/piper"

    driver = _D.__new__(_D)
    driver._managed_launch = instance
    BaseROS2Driver.connect_to_device(driver)
    instance.start.assert_called_once()
    instance.wait_ready.assert_called_once()


def test_configure_creates_managed_launch_from_manifest() -> None:
    with patch("cyberwave.driver.ros2.managed_launch.ManagedRosLaunch") as ML:
        instance = MagicMock()
        ML.return_value = instance

        class _D(BaseROS2Driver):
            REGISTRY_ID = "test/piper"

        driver = _D.__new__(_D)
        driver._manifest = MagicMock(
            managed_launch=ManifestManagedLaunch(
                package="piper",
                launch_file="start_single_piper.launch.py",
                readiness=ManifestReadiness(name="/enable_srv"),
            )
        )
        driver._managed_launch = None
        BaseROS2Driver.configure(driver)
        ML.assert_called_once()
        assert driver._managed_launch is instance


def test_build_shell_command_with_package() -> None:
    from cyberwave.driver.ros2.managed_launch import ManagedRosLaunch

    spec = ManifestManagedLaunch(
        package="piper",
        launch_file="start_single_piper.launch.py",
        ros_setup="/opt/ros/humble/setup.bash",
    )
    cmd = ManagedRosLaunch(spec).build_shell_command()
    assert "ros2 launch piper start_single_piper.launch.py" in cmd


def test_build_shell_command_empty_package_runs_launch_file_as_path() -> None:
    from cyberwave.driver.ros2.managed_launch import ManagedRosLaunch

    spec = ManifestManagedLaunch(
        package="",
        launch_file="/app/start_single_piper_namespaced.launch.py",
        launch_args={"can_port": "can1", "cw_namespace": "/CW_ABC"},
        ros_setup="/opt/ros/humble/setup.bash",
    )
    cmd = ManagedRosLaunch(spec).build_shell_command()
    # No empty package token; launch_file invoked directly as a path.
    assert "ros2 launch /app/start_single_piper_namespaced.launch.py" in cmd
    assert "ros2 launch  " not in cmd
    assert "ros2 launch '' " not in cmd
    assert "cw_namespace:=/CW_ABC" in cmd
