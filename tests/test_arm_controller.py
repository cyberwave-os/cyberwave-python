"""ArmController: pure joint-target math with fake kinematics (no pinocchio)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cyberwave.driver.kinematics import ArmKinematicsConfig, GripperConfig
from cyberwave.driver.kinematics.arm.controller import ArmController, pose_to_matrix


@dataclass
class _Pose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


class _FakeKin:
    """Deterministic stand-in for BaseKinematicsManipulator; records ik() calls."""

    def __init__(self, config: ArmKinematicsConfig, *, solve: bool = True) -> None:
        self.config = config
        self.solve = solve
        self.ik_calls: list[tuple[dict, np.ndarray]] = []

    def fk(self, positions):
        return np.eye(4)

    def apply_translation(self, pose, components, *, frame="tool"):
        return np.eye(4)

    def apply_rotation(self, pose, axis, angle, *, frame="tool"):
        return np.eye(4)

    def ik(self, start, target):
        self.ik_calls.append((dict(start), np.asarray(target)))
        return {n: 0.25 for n in self.config.arm_joints} if self.solve else None

    def joint_limits(self):
        return {n: (-1.0, 1.0) for n in self.config.arm_joints}


def _cfg() -> ArmKinematicsConfig:
    return ArmKinematicsConfig(
        urdf_path="/x.urdf",
        ee_frame="tool",
        arm_joints=("joint1", "joint2"),
        tool_axes={"forward": np.array([0.0, 0.0, 1.0])},
        rot_axes={
            "yaw": np.array([-1.0, 0.0, 0.0]),
            "pitch": np.array([0.0, -1.0, 0.0]),
        },
        gripper=GripperConfig(
            names=("jg",), open=0.0, closed=0.04, mimic={"jg_mimic": -1.0}
        ),
        base_joint="joint1",
    )


def _controller(*, solve: bool = True) -> ArmController:
    return ArmController(_cfg(), build_kinematics=lambda c: _FakeKin(c, solve=solve))


def test_translate_solves_arm_joints() -> None:
    ctrl = _controller()
    sol = ctrl.translate({"joint1": 0.5, "joint2": 0.0}, {"forward": 0.1})
    assert sol == {"joint1": 0.25, "joint2": 0.25}


def test_translate_returns_none_on_ik_miss() -> None:
    ctrl = _controller(solve=False)
    assert ctrl.translate({"joint1": 0.0}, {"forward": 0.1}) is None


def test_rotate_solves_arm_joints() -> None:
    ctrl = _controller()
    sol = ctrl.rotate({"joint1": 0.0, "joint2": 0.0}, "yaw", 0.2)
    assert sol == {"joint1": 0.25, "joint2": 0.25}


def test_base_rotate_clamps_to_limits() -> None:
    ctrl = _controller()
    # current joint1 = 0.5; +1.0 rad -> clamp to upper limit 1.0
    assert ctrl.base_rotate({"joint1": 0.5}, 1.0) == {"joint1": 1.0}


def test_pose_to_joints_builds_target_and_solves() -> None:
    kin = _FakeKin(_cfg())
    ctrl = ArmController(_cfg(), build_kinematics=lambda c: kin)
    sol = ctrl.pose_to_joints({"joint1": 0.0, "joint2": 0.0}, _Pose(x=0.1, yaw=0.5))
    assert sol == {"joint1": 0.25, "joint2": 0.25}
    # target passed to ik is the pose matrix (translation column carries x,y,z)
    _, target = kin.ik_calls[-1]
    assert target[0, 3] == 0.1


def test_pose_to_joints_returns_none_on_ik_miss() -> None:
    ctrl = _controller(solve=False)
    assert ctrl.pose_to_joints({"joint1": 0.0}, _Pose(x=0.1)) is None


def test_gripper_target_maps_force_and_mimics() -> None:
    ctrl = _controller()
    assert ctrl.gripper_target(0.5) == {"jg": 0.02, "jg_mimic": -0.02}


def test_gripper_target_empty_without_gripper() -> None:
    cfg = ArmKinematicsConfig(urdf_path="", ee_frame="", arm_joints=())
    ctrl = ArmController(cfg, build_kinematics=lambda c: _FakeKin(c))
    assert ctrl.gripper_target(1.0) == {}


def test_classify_gripper_ties_go_open() -> None:
    ctrl = _controller()
    assert ctrl.classify_gripper(0.02) == "open"  # tie -> open
    assert ctrl.classify_gripper(0.03) == "closed"


def test_binarize_snaps_gripper_and_present_mimic() -> None:
    ctrl = _controller()
    out = ctrl.binarize({"joint1": 0.5, "jg": 0.03, "jg_mimic": -0.03})
    assert out["joint1"] == 0.5
    assert out["jg"] == 0.04
    assert out["jg_mimic"] == -0.04


def test_hold_defaults_to_open_commanded_names_only() -> None:
    ctrl = _controller()
    assert ctrl.hold(None) == {"jg": 0.0}
    assert ctrl.hold("closed") == {"jg": 0.04}


def test_pose_to_matrix_translation_and_identity_rotation() -> None:
    m = pose_to_matrix(_Pose(x=1.0, y=2.0, z=3.0))
    assert np.allclose(m[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(m[:3, :3], np.eye(3))
    assert np.allclose(m[3], [0.0, 0.0, 0.0, 1.0])
