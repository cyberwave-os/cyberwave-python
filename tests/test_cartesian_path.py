"""Straight-line Cartesian waypoint generation on the simple_arm URDF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    import pinocchio  # noqa: F401
except ImportError:
    pytest.skip("pinocchio unavailable", allow_module_level=True)

from cyberwave.driver.kinematics import ArmKinematicsConfig, BaseKinematicsManipulator  # noqa: E402

_URDF = str(Path(__file__).resolve().parent / "fixtures" / "simple_arm.urdf")


def _kin() -> BaseKinematicsManipulator:
    return BaseKinematicsManipulator(
        ArmKinematicsConfig(
            urdf_path=_URDF,
            ee_frame="tool_link",
            arm_joints=("joint1", "joint2"),
            locked_joints=("gripper_joint",),
            tool_axes={"forward": np.array([0.0, 0.0, 1.0])},
            base_axes={"forward": np.array([0.0, 0.0, 1.0])},
            rot_axes={"yaw": np.array([0.0, 0.0, 1.0])},
        )
    )


def _pose(x: float, y: float, z: float) -> np.ndarray:
    out = np.eye(4)
    out[:3, 3] = [x, y, z]
    return out


def test_interpolate_endpoints_are_exact() -> None:
    kin = _kin()
    a, b = _pose(0.0, 0.0, 0.4), _pose(0.1, 0.0, 0.2)
    assert kin.interpolate_pose(a, b, 0.0) == pytest.approx(a)
    assert kin.interpolate_pose(a, b, 1.0) == pytest.approx(b)


def test_translation_midpoint_is_a_straight_line() -> None:
    """The whole point of the task: screw interpolation would curve this."""
    kin = _kin()
    a, b = _pose(0.0, 0.0, 0.4), _pose(0.1, 0.0, 0.2)
    mid = kin.interpolate_pose(a, b, 0.5)
    assert mid[:3, 3] == pytest.approx([0.05, 0.0, 0.3])


def test_rotation_is_slerped() -> None:
    import pinocchio as pin

    kin = _kin()
    a = np.eye(4)
    b = np.eye(4)
    b[:3, :3] = pin.utils.rpyToMatrix(0.0, 0.4, 0.0)
    mid = kin.interpolate_pose(a, b, 0.5)
    assert pin.rpy.matrixToRpy(mid[:3, :3]) == pytest.approx([0.0, 0.2, 0.0], abs=1e-9)


def test_pose_waypoints_count_and_final_pose() -> None:
    kin = _kin()
    a, b = _pose(0.0, 0.0, 0.4), _pose(0.0, 0.0, 0.2)
    wps = kin.pose_waypoints(a, b, 4)
    assert len(wps) == 4
    assert wps[-1] == pytest.approx(b)
    zs = [w[2, 3] for w in wps]
    assert zs == pytest.approx([0.35, 0.30, 0.25, 0.20])


def test_pose_waypoints_single_step_is_the_target() -> None:
    kin = _kin()
    a, b = _pose(0.0, 0.0, 0.4), _pose(0.0, 0.0, 0.2)
    assert kin.pose_waypoints(a, b, 1) == [pytest.approx(b)]


def test_pose_waypoints_rejects_non_positive_steps() -> None:
    kin = _kin()
    with pytest.raises(ValueError, match="steps must be >= 1"):
        kin.pose_waypoints(np.eye(4), np.eye(4), 0)
