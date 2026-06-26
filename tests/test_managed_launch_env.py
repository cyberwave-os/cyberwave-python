from __future__ import annotations

import os
from unittest.mock import patch

from cyberwave.driver.ros2.env_params import resolve_managed_launch_args
from cyberwave.driver.ros2.manifest import ManifestManagedLaunch, ManifestParam, NodeManifest


def test_resolve_managed_launch_args_merges_cw_ros2_env() -> None:
    manifest = NodeManifest(
        params=[ManifestParam(name="can_port", type="string", default_value="can0")],
        managed_launch=ManifestManagedLaunch(
            package="piper",
            launch_file="start_single_piper.launch.py",
            launch_args={"can_port": "can0", "auto_enable": False},
        ),
    )
    with patch.dict(os.environ, {"CW_ROS2_CAN_PORT": "can1"}):
        args = resolve_managed_launch_args(manifest, manifest.managed_launch)
    assert args["can_port"] == "can1"
    assert args["auto_enable"] is False


def test_managed_launch_build_includes_overlay() -> None:
    from cyberwave.driver.ros2.managed_launch import ManagedRosLaunch

    spec = ManifestManagedLaunch(
        package="piper",
        launch_file="start_single_piper.launch.py",
        ros_setup="/opt/ros/humble/setup.bash",
    )
    with patch.dict(
        os.environ,
        {"ROS_SETUP_OVERLAY": "/ws/piper_ros/install/setup.bash"},
    ):
        cmd = ManagedRosLaunch(spec).build_shell_command()
    assert "/opt/ros/humble/setup.bash" in cmd
    assert "/ws/piper_ros/install/setup.bash" in cmd
