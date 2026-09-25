"""EECartesianPoseMixin: ee_pose registration + IK-off/IK-on dispatch (no pinocchio)."""

from __future__ import annotations

import numpy as np

from cyberwave.driver import (
    ArmCapabilityMixin,
    ArmKinematicsConfig,
    EECartesianPoseMixin,
    EEPose,
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
        return {n: 0.25 for n in self.config.arm_joints} if self.solve else None

    def joint_limits(self):
        return {n: (-1.0, 1.0) for n in self.config.arm_joints}


def _cfg() -> ArmKinematicsConfig:
    return ArmKinematicsConfig(
        urdf_path="/x.urdf",
        ee_frame="tool",
        arm_joints=("joint1", "joint2"),
        tool_axes={"forward": np.array([0.0, 0.0, 1.0])},
        rot_axes={"yaw": np.array([-1.0, 0.0, 0.0])},
    )


class _TerminalBase:
    def define_interface(self, iface):
        pass


class _Driver(EECartesianPoseMixin, ArmCapabilityMixin, _TerminalBase):
    use_base_commands = True
    use_cartesian_pose = True

    def __init__(self, *, process_joints_ik=False, solve=True):
        super().__init__(process_joints_ik=process_joints_ik)
        self._cfg = _cfg()
        self._solve = solve
        self.applied: list[dict[str, float]] = []
        self.pose_calls: list[tuple[EEPose, str | None]] = []
        self.failures: list[tuple[str | None, str]] = []
        self._joints = {"joint1": 0.0, "joint2": 0.0}

    def arm_config(self):
        return self._cfg

    def _apply_joint_targets(self, positions):
        self.applied.append(dict(positions))

    def _current_joint_positions(self):
        return dict(self._joints)

    def _on_arm_command_failed(self, command, reason):
        self.failures.append((command, reason))

    def _build_arm_kinematics(self, config):
        return _FakeKin(config, solve=self._solve)

    def _apply_ee_pose(self, pose, *, arm=None):
        self.pose_calls.append((pose, arm))


def _supported_commands(driver) -> list:
    iface = DriverInterfaceRegistry()
    driver.define_interface(iface)
    return iface.to_cw_driver_dict(registry_id="test")["mqtt"]["commands"]["supported"]


def _command_names(driver) -> set[str]:
    iface = DriverInterfaceRegistry()
    driver.define_interface(iface)
    return set(iface.command_dispatch_table(DriverOperationMode.TELEOP_LOCAL).keys())


def test_from_payload_parses_6d_array() -> None:
    pose = EEPose.from_payload({"pose": [1.0, 0.0, 0.0, 0.0, 0.0, 0.5]})
    assert (pose.x, pose.y, pose.z) == (1.0, 0.0, 0.0)
    assert (pose.roll, pose.pitch, pose.yaw) == (0.0, 0.0, 0.5)


def test_from_payload_rejects_wrong_length() -> None:
    import pytest

    with pytest.raises(ValueError):
        EEPose.from_payload({"pose": [1.0, 2.0, 3.0]})


def test_ee_pose_registered_when_enabled() -> None:
    assert "ee_pose" in _command_names(_Driver())


def test_ee_pose_not_registered_when_disabled() -> None:
    class Off(_Driver):
        use_cartesian_pose = False

    assert "ee_pose" not in _command_names(Off())


def test_wire_payload_carries_only_pose_fields() -> None:
    supported = _supported_commands(_Driver())
    (entry,) = [c for c in supported if isinstance(c, dict) and c["name"] == "ee_pose"]
    arg_names = [a["name"] for a in entry["args"]]
    assert arg_names == ["pose"]  # single 6-element array, not per-field args
    assert "mode1" not in arg_names and "mode2" not in arg_names


def test_ik_off_forwards_raw_pose_to_apply_ee_pose() -> None:
    driver = _Driver(process_joints_ik=False)
    driver._on_ee_pose(
        {"command": "ee_pose", "data": {"pose": [0.3, 0.0, 0.0, 0.0, 0.0, 0.1]}}
    )
    assert driver.applied == []
    (pose, arm) = driver.pose_calls[-1]
    assert (pose.x, pose.yaw, arm) == (0.3, 0.1, None)


def test_ik_off_default_stub_surfaces_failure() -> None:
    class Stub(_Driver):
        def _apply_ee_pose(self, pose, *, arm=None):
            raise NotImplementedError("no ee_pose send")

    driver = Stub(process_joints_ik=False)
    driver._on_ee_pose(
        {"command": "ee_pose", "data": {"pose": [0.3, 0.0, 0.0, 0.0, 0.0, 0.0]}}
    )
    assert driver.applied == []
    assert driver.failures and driver.failures[-1][0] == "ee_pose"


def test_ik_on_produces_joint_targets() -> None:
    driver = _Driver(process_joints_ik=True, solve=True)
    driver._on_ee_pose(
        {"command": "ee_pose", "data": {"pose": [0.3, 0.0, 0.0, 0.0, 0.0, 0.0]}}
    )
    assert driver.applied[-1] == {"joint1": 0.25, "joint2": 0.25}
    assert driver.pose_calls == []


def test_ik_on_miss_surfaces_failure() -> None:
    driver = _Driver(process_joints_ik=True, solve=False)
    driver._on_ee_pose(
        {"command": "ee_pose", "data": {"pose": [0.3, 0.0, 0.0, 0.0, 0.0, 0.0]}}
    )
    assert driver.applied == []
    assert driver.failures and driver.failures[-1][0] == "ee_pose"
