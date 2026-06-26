from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from cyberwave.driver.ros2.manifest import ManifestManagedLaunch
from cyberwave.driver.ros2.ros_setup_env import (
    apply_ros_setup_environment,
    collect_ros_setup_scripts,
)


def test_collect_ros_setup_scripts_manifest_and_overlay_env() -> None:
    spec = ManifestManagedLaunch(
        package="piper",
        launch_file="start_single_piper.launch.py",
        ros_setup="/opt/ros/humble/setup.bash",
    )
    with patch.dict(
        os.environ,
        {"ROS_SETUP_OVERLAY": "/ws/piper_ros/install/setup.bash"},
    ):
        scripts = collect_ros_setup_scripts(spec)
    assert scripts == [
        "/opt/ros/humble/setup.bash",
        "/ws/piper_ros/install/setup.bash",
    ]


def test_bootstrap_ros_python_process_reexec_when_import_fails(tmp_path) -> None:
    import cyberwave.driver.ros2.ros_setup_env as rse

    setup = tmp_path / "setup.bash"
    setup.write_text("# stub\n")
    scripts = [str(setup)]
    with (
        patch.object(rse, "apply_ros_setup_environment") as apply_env,
        patch.object(
            rse, "_ros_python_executable", return_value="/usr/bin/python3.10"
        ),
        patch.object(rse.importlib, "import_module") as imp,
        patch.object(rse.os, "execve") as execve,
        patch.object(rse.sys, "executable", "/usr/bin/python3.13"),
        patch.dict(os.environ, {}, clear=True),
    ):
        imp.side_effect = ImportError("piper_msgs")
        rse.bootstrap_ros_python_process(
            scripts,
            gate_import="piper_msgs",
            argv=["/app/main.py"],
        )
        apply_env.assert_called_once_with(scripts)
        execve.assert_called_once()
        assert os.environ["CW_ROS_PYTHON_READY"] == "1"


def test_apply_ros_setup_environment_merges_pythonpath(tmp_path) -> None:
    setup = tmp_path / "setup.bash"
    setup.write_text("# stub\n")
    with patch("cyberwave.driver.ros2.ros_setup_env.subprocess.run") as run:
        run.return_value = MagicMock(
            stdout="PYTHONPATH=/ws/install/lib/python3.10/site-packages\nROS_DISTRO=humble\n",
            returncode=0,
        )
        apply_ros_setup_environment([str(setup)])
    assert os.environ["PYTHONPATH"] == "/ws/install/lib/python3.10/site-packages"
    assert os.environ["ROS_DISTRO"] == "humble"
