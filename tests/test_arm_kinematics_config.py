"""ArmKinematicsConfig: pure dataclass + command-gating flags (no pinocchio)."""

from __future__ import annotations

import numpy as np

from cyberwave.driver.kinematics import ArmKinematicsConfig, GripperConfig


def _ee_axes() -> dict[str, np.ndarray]:
    return {
        "forward": np.array([0.0, 0.0, 1.0]),
        "left": np.array([0.0, 1.0, 0.0]),
        "up": np.array([-1.0, 0.0, 0.0]),
    }


def test_minimal_config_supports_nothing() -> None:
    cfg = ArmKinematicsConfig(urdf_path="", ee_frame="", arm_joints=())
    assert cfg.supports_ee is False
    assert cfg.supports_gripper is False
    assert cfg.supports_base_rotate is False
    assert cfg.supports_home is True  # home defaults to all-zero pose


def test_ee_gating_requires_frame_and_axes() -> None:
    cfg = ArmKinematicsConfig(
        urdf_path="/x.urdf",
        ee_frame="tool",
        arm_joints=("j1", "j2"),
        tool_axes=_ee_axes(),
        rot_axes={"yaw": np.array([-1.0, 0.0, 0.0])},
    )
    assert cfg.supports_ee is True


def test_gripper_gating_requires_gripper_config() -> None:
    cfg = ArmKinematicsConfig(
        urdf_path="/x.urdf", ee_frame="tool", arm_joints=("j1",),
        gripper=GripperConfig(names=("jg",), open=0.0, closed=0.04),
    )
    assert cfg.supports_gripper is True


def test_no_gripper_config_disables_gripper() -> None:
    cfg = ArmKinematicsConfig(urdf_path="", ee_frame="", arm_joints=())
    assert cfg.supports_gripper is False


def test_gripper_config_mimic_is_immutable() -> None:
    g = GripperConfig(names=("jg",), mimic={"jg_m": -1.0})
    assert dict(g.mimic) == {"jg_m": -1.0}
    import pytest
    with pytest.raises(TypeError):
        g.mimic["x"] = 1.0  # frozen/immutable view


def test_base_rotate_gating_requires_base_joint() -> None:
    cfg = ArmKinematicsConfig(
        urdf_path="/x.urdf", ee_frame="tool", arm_joints=("j1",), base_joint="j1"
    )
    assert cfg.supports_base_rotate is True


def test_home_gating_true_by_default() -> None:
    cfg = ArmKinematicsConfig(urdf_path="", ee_frame="", arm_joints=())
    assert cfg.supports_home is True


def test_config_is_frozen() -> None:
    import pytest

    cfg = ArmKinematicsConfig(urdf_path="", ee_frame="", arm_joints=("j1",))
    with pytest.raises(Exception):
        cfg.ee_frame = "changed"  # type: ignore[misc]
