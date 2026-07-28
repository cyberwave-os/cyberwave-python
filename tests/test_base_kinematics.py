"""BaseKinematicsManipulator FK/IK/limits on the synthetic simple_arm URDF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Skip when pinocchio can't be imported for ANY reason (native libs may be
# absent in CI); pytest>=8.2 narrowed importorskip() to ModuleNotFoundError.
try:
    import pinocchio  # noqa: F401
except ImportError:
    pytest.skip("pinocchio unavailable", allow_module_level=True)

from cyberwave.driver.kinematics import ArmKinematicsConfig, BaseKinematicsManipulator  # noqa: E402

_URDF = str(Path(__file__).resolve().parent / "fixtures" / "simple_arm.urdf")


def _cfg() -> ArmKinematicsConfig:
    return ArmKinematicsConfig(
        urdf_path=_URDF,
        ee_frame="tool_link",
        arm_joints=("joint1", "joint2"),
        locked_joints=("gripper_joint",),
        tool_axes={
            "forward": np.array([0.0, 0.0, 1.0]),
            "left": np.array([0.0, 1.0, 0.0]),
            "up": np.array([-1.0, 0.0, 0.0]),
        },
        base_axes={
            "forward": np.array([0.0, 0.0, 1.0]),
            "up": np.array([0.0, 0.0, 1.0]),
            "left": np.array([0.0, 1.0, 0.0]),
        },
        rot_axes={
            "yaw": np.array([-1.0, 0.0, 0.0]),
            "pitch": np.array([0.0, -1.0, 0.0]),
        },
    )


def _kin() -> BaseKinematicsManipulator:
    return BaseKinematicsManipulator(_cfg())


def test_fk_returns_pose_for_zero_config() -> None:
    pose = _kin().fk({"joint1": 0.0, "joint2": 0.0})
    assert pose.shape == (4, 4)
    # tool_link sits at z = 0.1 + 0.2 + 0.15 = 0.45 above base at zero config.
    assert pose[2, 3] == pytest.approx(0.45, abs=1e-6)


def test_arm_joint_limits_match_urdf() -> None:
    limits = _kin().joint_limits()
    assert limits["joint1"] == pytest.approx((-2.618, 2.618), abs=1e-3)
    assert limits["joint2"] == pytest.approx((-1.57, 1.57), abs=1e-3)


def test_velocity_limits_from_urdf():
    kin = BaseKinematicsManipulator(_cfg())
    limits = kin.velocity_limits()
    assert limits == {"joint1": pytest.approx(1.0), "joint2": pytest.approx(1.0)}


def test_ik_round_trips_a_reachable_pose() -> None:
    kin = _kin()
    seed = {"joint1": 0.3, "joint2": 0.5}
    target = kin.fk(seed)
    solution = kin.ik({"joint1": 0.0, "joint2": 0.0}, target)
    assert solution is not None
    reached = kin.fk(solution)
    assert np.allclose(reached[:3, 3], target[:3, 3], atol=1e-3)


def test_ik_returns_none_for_unreachable_target() -> None:
    kin = _kin()
    target = kin.fk({"joint1": 0.0, "joint2": 0.0})
    target[0, 3] += 5.0  # 5 m away — far outside the workspace
    assert kin.ik({"joint1": 0.0, "joint2": 0.0}, target) is None


def test_apply_translation_tool_frame_moves_along_approach_axis() -> None:
    kin = _kin()
    pose = kin.fk({"joint1": 0.0, "joint2": 0.0})
    moved = kin.apply_translation(pose, {"forward": 0.1}, frame="tool")
    approach = pose[:3, 2]  # tool +Z column
    delta = moved[:3, 3] - pose[:3, 3]
    assert np.allclose(delta, 0.1 * approach, atol=1e-9)


def test_apply_rotation_changes_orientation_not_position() -> None:
    kin = _kin()
    pose = kin.fk({"joint1": 0.0, "joint2": 0.0})
    rotated = kin.apply_rotation(pose, "yaw", 0.5, frame="tool")
    assert np.allclose(rotated[:3, 3], pose[:3, 3], atol=1e-9)
    assert not np.allclose(rotated[:3, :3], pose[:3, :3])


def test_missing_pinocchio_message(monkeypatch) -> None:
    # Simulate absent pinocchio → clear install hint mentioning the extra.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pinocchio":
            raise ImportError("No module named 'pinocchio'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"cyberwave\[drivers\]"):
        BaseKinematicsManipulator(_cfg())
