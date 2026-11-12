"""ArmController: pure joint-target math with fake kinematics (no pinocchio)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

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

    def pose_waypoints(self, start, target, steps):
        return [np.asarray(target) for _ in range(steps)]


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


def test_loaded_kinematics_never_builds_the_model() -> None:
    """A status read must not pay for (or trigger) the kinematics build."""
    built: list[object] = []

    def build(cfg):
        built.append(cfg)
        return _FakeKin(cfg)

    ctrl = ArmController(_cfg(), build_kinematics=build)
    assert ctrl.loaded_kinematics is None
    assert built == []
    assert ctrl.loaded_kinematics is None
    kin = ctrl.kinematics
    assert ctrl.loaded_kinematics is kin
    assert len(built) == 1


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


def test_pose_path_single_step_returns_one_solution() -> None:
    ctrl = _controller()
    path = ctrl.pose_path_to_joints({"joint1": 0.0, "joint2": 0.0}, _Pose(z=0.3))
    assert path is not None
    assert len(path) == 1
    assert path[0] == {"joint1": 0.25, "joint2": 0.25}


def test_pose_path_seeds_each_waypoint_from_the_previous_solution() -> None:
    """The continuity chain: waypoint N is solved from waypoint N-1's answer."""
    kin_holder: list[_FakeKin] = []

    def _build(cfg):
        kin = _FakeKin(cfg)
        kin_holder.append(kin)
        return kin

    ctrl = ArmController(_cfg(), build_kinematics=_build)
    path = ctrl.pose_path_to_joints(
        {"joint1": 0.0, "joint2": 0.0}, _Pose(z=0.3), steps=3
    )
    assert path is not None and len(path) == 3

    starts = [call[0] for call in kin_holder[0].ik_calls]
    assert starts[0] == {"joint1": 0.0, "joint2": 0.0}
    assert starts[1] == {"joint1": 0.25, "joint2": 0.25}
    assert starts[2] == {"joint1": 0.25, "joint2": 0.25}


def test_pose_path_returns_none_when_any_waypoint_misses() -> None:
    ctrl = _controller(solve=False)
    assert ctrl.pose_path_to_joints({"joint1": 0.0}, _Pose(z=0.3), steps=3) is None


def test_pose_path_rejects_a_discontinuous_jump() -> None:
    class _JumpyKin(_FakeKin):
        def __init__(self, config):
            super().__init__(config)
            self._n = 0

        def ik(self, start, target):
            self.ik_calls.append((dict(start), np.asarray(target)))
            self._n += 1
            value = 0.1 if self._n < 3 else 3.0  # third waypoint flips branch
            return {n: value for n in self.config.arm_joints}

    ctrl = ArmController(_cfg(), build_kinematics=_JumpyKin)
    assert (
        ctrl.pose_path_to_joints(
            {"joint1": 0.0, "joint2": 0.0}, _Pose(z=0.3), steps=3, max_joint_step=0.5
        )
        is None
    )


def test_pose_path_allows_the_jump_when_unconstrained() -> None:
    class _JumpyKin(_FakeKin):
        def ik(self, start, target):
            self.ik_calls.append((dict(start), np.asarray(target)))
            return {n: 3.0 for n in self.config.arm_joints}

    ctrl = ArmController(_cfg(), build_kinematics=_JumpyKin)
    path = ctrl.pose_path_to_joints({"joint1": 0.0}, _Pose(z=0.3), steps=2)
    assert path is not None and len(path) == 2


def test_pose_matrix_path_can_cross_an_explicit_interior_ik_gap() -> None:
    class _GapKin(_FakeKin):
        last_ik_refusal = None

        def ik(self, start, target):
            x = float(target[0, 3])
            self.ik_calls.append((dict(start), np.asarray(target)))
            if x == 1.0:
                return None
            return {name: x for name in self.config.arm_joints}

    poses = []
    for x in (0.5, 1.0, 1.5):
        pose = np.eye(4)
        pose[0, 3] = x
        poses.append(pose)
    ctrl = ArmController(_cfg(), build_kinematics=_GapKin)

    path = ctrl.pose_matrix_path_to_joints(
        {"joint1": 0.0, "joint2": 0.0}, poses, allow_ik_gaps=True
    )

    assert path == [
        {"joint1": 0.5, "joint2": 0.5},
        {"joint1": 1.5, "joint2": 1.5},
    ]


def test_pose_matrix_path_never_skips_a_collision_refusal() -> None:
    class _CollisionKin(_FakeKin):
        last_ik_refusal = "self-collision"

        def ik(self, start, target):
            return None

    ctrl = ArmController(_cfg(), build_kinematics=_CollisionKin)

    assert (
        ctrl.pose_matrix_path_to_joints(
            {"joint1": 0.0, "joint2": 0.0},
            [np.eye(4), np.eye(4)],
            allow_ik_gaps=True,
        )
        is None
    )


def test_pose_matrix_path_never_skips_an_unresolved_final_pose() -> None:
    ctrl = _controller(solve=False)

    assert (
        ctrl.pose_matrix_path_to_joints(
            {"joint1": 0.0, "joint2": 0.0},
            [np.eye(4), np.eye(4)],
            allow_ik_gaps=True,
        )
        is None
    )


def test_the_reported_orientation_of_identity_is_unit_w() -> None:
    from cyberwave.driver.motion.poses import transform_to_pose

    assert transform_to_pose(np.eye(4)).orientation == pytest.approx(
        (0.0, 0.0, 0.0, 1.0)
    )


def test_the_reported_orientation_of_a_half_turn_about_z() -> None:
    from cyberwave.driver.motion.poses import transform_to_pose

    pose = np.eye(4)
    pose[:3, :3] = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    x, y, z, w = transform_to_pose(pose).orientation
    assert (x, y, abs(z), abs(w)) == pytest.approx((0.0, 0.0, 1.0, 0.0), abs=1e-9)


def test_binarize_measured_false_preserves_continuous_width() -> None:
    cfg = ArmKinematicsConfig(
        urdf_path="/x.urdf",
        ee_frame="tool",
        arm_joints=("joint1",),
        tool_axes={"forward": np.array([0.0, 0.0, 1.0])},
        rot_axes={"yaw": np.array([0.0, 0.0, 1.0])},
        gripper=GripperConfig(
            names=("jg",),
            open=0.0,
            closed=0.04,
            mimic={"jg_mimic": -1.0},
            binarize=True,
            binarize_measured=False,
        ),
    )
    ctrl = ArmController(cfg, build_kinematics=lambda c: _FakeKin(c))
    assert ctrl.binarize({"jg": 0.021}) == {"jg": 0.021}
    # The command path still snaps.
    assert ctrl.gripper_target(1.0)["jg"] == 0.04


def test_binarize_measured_defaults_to_snapping() -> None:
    ctrl = _controller()
    assert ctrl.binarize({"jg": 0.039})["jg"] == 0.04


def test_kinematics_accessor_exposes_the_single_instance() -> None:
    """One model per controller: the state publisher must borrow, not rebuild."""
    ctrl = _controller()
    assert ctrl.kinematics is ctrl.kinematics
    assert isinstance(ctrl.kinematics, _FakeKin)


def test_ee_pose_reports_the_fk_pose_of_the_current_state() -> None:
    ctrl = _controller()
    pose = ctrl.ee_pose({"joint1": 0.1, "joint2": 0.2})
    assert pose.shape == (4, 4)
    assert pose == pytest.approx(np.eye(4))  # _FakeKin.fk


class _RestartingKin(_FakeKin):
    """A multi-seed solver: records the ``restart`` each solve was given."""

    def __init__(self, config: ArmKinematicsConfig) -> None:
        super().__init__(config)
        self.restarts: list[bool] = []

    def ik(self, start, target, *, restart=True):
        self.restarts.append(restart)
        return super().ik(start, target)


def test_jogs_stay_on_the_current_ik_branch() -> None:
    # A multi-seed restart answers a 1 cm jog with another posture reaching
    # the same point; the arm would swing through it. Jogs solve single-seed.
    kin: list[_RestartingKin] = []
    ctrl = ArmController(
        _cfg(), build_kinematics=lambda c: kin.append(_RestartingKin(c)) or kin[0]
    )
    ctrl.translate({"joint1": 0.5, "joint2": 0.0}, {"forward": 0.01})
    ctrl.rotate({"joint1": 0.5, "joint2": 0.0}, "yaw", 0.05)
    assert kin[0].restarts == [False, False]
