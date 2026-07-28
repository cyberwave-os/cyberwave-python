"""DualArmMixin: prefixed command routing, conjunct home, unified surface (no pinocchio)."""

from __future__ import annotations

import numpy as np

from cyberwave.driver import (
    ArmKinematicsConfig,
    DualArmConfig,
    DualArmMixin,
    EECartesianPoseMixin,
    EEPose,
    GripperConfig,
)
from cyberwave.driver.interface import DriverOperationMode
from cyberwave.driver.interface.registry import DriverInterfaceRegistry


class _FakeKin:
    def __init__(self, config, *, solve=True):
        self.config = config
        self.solve = solve

    def fk(self, positions):
        return np.eye(4)

    def apply_translation(self, pose, components, *, frame="tool"):
        return np.eye(4)

    def apply_rotation(self, pose, axis, angle, *, frame="tool"):
        return np.eye(4)

    def ik(self, start, target):
        # Echo prefixed arm-joint names so the target disambiguates the arm.
        return {n: 0.25 for n in self.config.arm_joints} if self.solve else None

    def joint_limits(self):
        return {n: (-1.0, 1.0) for n in self.config.arm_joints}


def _arm_cfg(prefix: str) -> ArmKinematicsConfig:
    return ArmKinematicsConfig(
        urdf_path="/openarm.urdf",
        ee_frame=f"{prefix}_tool",
        arm_joints=(f"{prefix}_joint1", f"{prefix}_joint2"),
        tool_axes={"forward": np.array([0.0, 0.0, 1.0])},
        rot_axes={"yaw": np.array([-1.0, 0.0, 0.0])},
        gripper=GripperConfig(names=(f"{prefix}_grip",), open=0.0, closed=0.04),
        base_joint=f"{prefix}_joint1",
    )


class _TerminalBase:
    def define_interface(self, iface):
        pass


class _Driver(DualArmMixin, EECartesianPoseMixin, _TerminalBase):
    use_dual_arm = True
    use_cartesian_pose = True

    def __init__(self, *, process_joints_ik=False):
        super().__init__(process_joints_ik=process_joints_ik)
        self._cfg = DualArmConfig(left=_arm_cfg("left"), right=_arm_cfg("right"))
        self.applied: list[dict[str, float]] = []
        self.pose_calls: list[tuple[EEPose, str | None]] = []
        self.failures: list[tuple[str | None, str]] = []
        self.homed = False
        self._joints = {
            "left_joint1": 0.5,
            "left_joint2": 0.0,
            "right_joint1": 0.5,
            "right_joint2": 0.0,
        }

    def dual_arm_config(self):
        return self._cfg

    def _apply_joint_targets(self, positions):
        self.applied.append(dict(positions))

    def _current_joint_positions(self):
        return dict(self._joints)

    def _on_arm_command_failed(self, command, reason):
        self.failures.append((command, reason))

    def _build_arm_kinematics(self, config):
        return _FakeKin(config)

    def request_home(self):
        self.homed = True

    def _apply_ee_pose(self, pose, *, arm=None):
        self.pose_calls.append((pose, arm))


def _command_names(driver) -> set[str]:
    iface = DriverInterfaceRegistry()
    driver.define_interface(iface)
    return set(iface.command_dispatch_table(DriverOperationMode.TELEOP_LOCAL).keys())


def test_registers_prefixed_and_conjunct_commands() -> None:
    names = _command_names(_Driver())
    assert {
        "left_ee_move",
        "right_ee_move",
        "left_ee_rotate_left",
        "right_ee_rotate_left",
        "left_base_rotate_left",
        "right_base_rotate_left",
        "left_grip",
        "right_grip",
        "left_release",
        "right_release",
        "left_home",
        "right_home",
        "left_ee_pose",
        "right_ee_pose",
        "home",
    } <= names
    assert "ee_pose" not in names  # bare ee_pose suppressed under dual-arm


def test_left_ee_move_routes_to_left_controller() -> None:
    driver = _Driver()
    driver._on_dual_arm_command({"command": "left_ee_move", "data": {"forward": 0.1}})
    assert driver.applied[-1] == {"left_joint1": 0.25, "left_joint2": 0.25}


def test_right_ee_move_routes_to_right_controller() -> None:
    driver = _Driver()
    driver._on_dual_arm_command({"command": "right_ee_move", "data": {"forward": 0.1}})
    assert driver.applied[-1] == {"right_joint1": 0.25, "right_joint2": 0.25}


def test_left_grip_uses_prefixed_gripper_joint() -> None:
    driver = _Driver()
    driver._on_dual_arm_command({"command": "left_grip", "data": {"force": 0.5}})
    assert driver.applied[-1] == {"left_grip": 0.02}


def test_left_ee_pose_passes_arm_label() -> None:
    driver = _Driver(process_joints_ik=False)
    driver._on_dual_arm_command(
        {"command": "left_ee_pose", "data": {"pose": [0.3, 0.0, 0.0, 0.0, 0.0, 0.0]}}
    )
    (pose, arm) = driver.pose_calls[-1]
    assert (pose.x, arm) == (0.3, "left")


def test_conjunct_home_fans_out_via_request_home() -> None:
    driver = _Driver()
    driver._on_dual_arm_command({"command": "home", "data": {}})
    assert driver.homed is True


def test_per_arm_home_stub_surfaces_failure() -> None:
    driver = _Driver()
    driver._on_dual_arm_command({"command": "left_home", "data": {}})
    assert driver.failures and driver.failures[-1][0] == "left_home"


def test_unknown_prefix_is_noop_warning() -> None:
    driver = _Driver()
    driver._on_dual_arm_command({"command": "middle_ee_move", "data": {}})
    assert driver.applied == []
