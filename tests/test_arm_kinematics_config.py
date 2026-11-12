"""ArmKinematicsConfig: pure dataclass + command-gating flags (no pinocchio)."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_tool_offset_defaults_to_none() -> None:
    cfg = ArmKinematicsConfig(urdf_path="", ee_frame="", arm_joints=())
    assert cfg.tool_offset is None


def test_tool_offset_from_xyz_rpy_is_homogeneous() -> None:
    from cyberwave.driver.kinematics import tool_offset_from_xyz_rpy

    m = tool_offset_from_xyz_rpy(0.0, 0.0, 0.1358)
    assert m.shape == (4, 4)
    assert m[:3, 3] == pytest.approx([0.0, 0.0, 0.1358])
    assert m[:3, :3] == pytest.approx(np.eye(3))
    assert m[3, :] == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_tool_offset_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="tool_offset must be a 4x4"):
        ArmKinematicsConfig(
            urdf_path="", ee_frame="", arm_joints=(), tool_offset=np.eye(3)
        )


def test_tool_offset_is_not_writable() -> None:
    cfg = ArmKinematicsConfig(
        urdf_path="", ee_frame="", arm_joints=(), tool_offset=np.eye(4)
    )
    assert cfg.tool_offset is not None
    with pytest.raises(ValueError):
        cfg.tool_offset[0, 3] = 1.0


def test_base_frame_defaults_to_the_urdf_root() -> None:
    # None: the kinematics resolve the URDF root link, so arms whose root is
    # not called base_link still build.
    cfg = ArmKinematicsConfig(urdf_path="", ee_frame="", arm_joints=())
    assert cfg.base_frame is None


def test_base_frame_is_overridable() -> None:
    cfg = ArmKinematicsConfig(
        urdf_path="", ee_frame="", arm_joints=(), base_frame="world"
    )
    assert cfg.base_frame == "world"


def _offset(rot: np.ndarray | None = None, bottom=(0.0, 0.0, 0.0, 1.0)) -> np.ndarray:
    m = np.eye(4)
    if rot is not None:
        m[:3, :3] = rot
    m[3, :] = bottom
    return m


def _cfg_with_offset(offset: np.ndarray) -> ArmKinematicsConfig:
    return ArmKinematicsConfig(
        urdf_path="", ee_frame="", arm_joints=(), tool_offset=offset
    )


def test_tool_offset_rejects_nan() -> None:
    bad = np.eye(4)
    bad[0, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _cfg_with_offset(bad)


def test_tool_offset_rejects_inf() -> None:
    bad = np.eye(4)
    bad[2, 3] = np.inf
    with pytest.raises(ValueError, match="finite"):
        _cfg_with_offset(bad)


def test_tool_offset_rejects_a_non_homogeneous_bottom_row() -> None:
    with pytest.raises(ValueError, match=r"bottom row"):
        _cfg_with_offset(_offset(bottom=(0.0, 0.0, 1.0, 1.0)))


def test_tool_offset_rejects_a_reflection() -> None:
    """det = -1: a mirrored frame is not a rigid transform."""
    flip = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="right-handed|rotation"):
        _cfg_with_offset(_offset(flip))


def test_tool_offset_rejects_a_singular_matrix() -> None:
    with pytest.raises(ValueError, match="rotation"):
        _cfg_with_offset(_offset(np.zeros((3, 3))))


def test_tool_offset_rejects_a_scaled_rotation() -> None:
    """Uniform scale is not rigid; it would silently rescale every distance."""
    with pytest.raises(ValueError, match="rotation"):
        _cfg_with_offset(_offset(2.0 * np.eye(3)))


def test_tool_offset_accepts_a_valid_rotation_and_translation() -> None:
    from cyberwave.driver.kinematics import tool_offset_from_xyz_rpy

    cfg = _cfg_with_offset(tool_offset_from_xyz_rpy(0.01, -0.02, 0.1358, 0.1, 0.2, 0.3))
    assert cfg.tool_offset is not None
    assert np.linalg.det(cfg.tool_offset[:3, :3]) == pytest.approx(1.0)


def test_tool_offset_helper_output_always_validates() -> None:
    """Whatever the helper produces must satisfy the config contract."""
    from cyberwave.driver.kinematics import tool_offset_from_xyz_rpy

    for rpy in ((0, 0, 0), (np.pi, 0, 0), (0, np.pi / 2, 0), (0.3, -1.2, 2.9)):
        _cfg_with_offset(tool_offset_from_xyz_rpy(0.1, 0.2, 0.3, *rpy))
