"""JointNameMap: rename both directions, mimic expansion, config derivation."""

from __future__ import annotations

import numpy as np
import pytest

from cyberwave.driver.ros2.joint_names import JointNameMap
from cyberwave.driver.kinematics import ArmKinematicsConfig, GripperConfig

PIPER_MAP = JointNameMap(
    ros_to_platform={"gripper": "joint7"},
    mimic={"joint8": ("joint7", -1.0)},
)


def test_to_platform_renames_and_expands_mimic():
    out = PIPER_MAP.to_platform(["joint1", "gripper"], [0.5, 0.03])
    assert out == {"joint1": 0.5, "joint7": 0.03, "joint8": -0.03}


def test_to_platform_no_mimic_when_source_absent():
    out = PIPER_MAP.to_platform(["joint1"], [0.5])
    assert out == {"joint1": 0.5}


def test_to_platform_identity_default():
    out = JointNameMap().to_platform(["a", "b"], [1.0, 2.0])
    assert out == {"a": 1.0, "b": 2.0}


def test_to_platform_zip_truncates_to_shortest():
    out = JointNameMap().to_platform(["a", "b"], [1.0])
    assert out == {"a": 1.0}


def test_to_ros_orders_and_renames():
    names, values = PIPER_MAP.to_ros(
        {"joint1": 0.1, "joint7": 0.02},
        order=("joint1", "joint2", "joint7"),
    )
    assert names == ["joint1", "joint2", "gripper"]
    assert values == [0.1, 0.0, 0.02]  # missing joint2 -> default 0.0


def test_from_arm_config_derives_mimic_from_gripper():
    cfg = ArmKinematicsConfig(
        urdf_path="/x.urdf",
        ee_frame="tool",
        arm_joints=("joint1",),
        tool_axes={"forward": np.array([0.0, 0.0, 1.0])},
        rot_axes={"yaw": np.array([1.0, 0.0, 0.0])},
        gripper=GripperConfig(
            names=("joint7",), open=0.035, closed=0.0, mimic={"joint8": -1.0}
        ),
    )
    nm = JointNameMap.from_arm_config(cfg, ros_to_platform={"gripper": "joint7"})
    assert nm.mimic == {"joint8": ("joint7", -1.0)}
    assert nm.to_platform(["gripper"], [0.02]) == {"joint7": 0.02, "joint8": -0.02}


def test_from_arm_config_without_gripper_has_no_mimic():
    cfg = ArmKinematicsConfig(
        urdf_path="/x.urdf",
        ee_frame="tool",
        arm_joints=("joint1",),
        tool_axes={"forward": np.array([0.0, 0.0, 1.0])},
        rot_axes={"yaw": np.array([1.0, 0.0, 0.0])},
    )
    assert JointNameMap.from_arm_config(cfg).mimic == {}
