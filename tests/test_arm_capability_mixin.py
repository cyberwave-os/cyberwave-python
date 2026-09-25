"""ArmCapabilityMixin: command gating + dispatch, no pinocchio, no ROS.

Kinematics is stubbed so these tests never import pinocchio; EE-solve behavior
is verified in test_base_kinematics.py.
"""

from __future__ import annotations

import numpy as np

from cyberwave.driver import ArmCapabilityMixin, ArmKinematicsConfig, GripperConfig
from cyberwave.driver.interface import DriverOperationMode
from cyberwave.driver.interface.registry import DriverInterfaceRegistry


class _FakeKin:
    """Stand-in for BaseKinematicsManipulator with deterministic outputs."""

    def __init__(self, config: ArmKinematicsConfig) -> None:
        self.config = config

    def fk(self, positions):
        return np.eye(4)

    def apply_translation(self, pose, components, *, frame="tool"):
        return np.eye(4)

    def apply_rotation(self, pose, axis, angle, *, frame="tool"):
        return np.eye(4)

    def ik(self, start, target):
        # Echo a fixed reachable solution for the arm joints.
        return {n: 0.25 for n in self.config.arm_joints}

    def joint_limits(self):
        return {n: (-1.0, 1.0) for n in self.config.arm_joints}


def _full_cfg() -> ArmKinematicsConfig:
    return ArmKinematicsConfig(
        urdf_path="/x.urdf",
        ee_frame="tool",
        arm_joints=("joint1", "joint2"),
        tool_axes={"forward": np.array([0.0, 0.0, 1.0])},
        rot_axes={"yaw": np.array([-1.0, 0.0, 0.0]), "pitch": np.array([0.0, -1.0, 0.0])},
        gripper=GripperConfig(names=("jg",), open=0.0, closed=0.04, mimic={"jg_mimic": -1.0}),
        base_joint="joint1",
    )


class _TerminalBase:
    """Stand-in for InterfaceRegistryMixin: terminates the super() chain."""

    def define_interface(self, iface):
        pass


class _Driver(ArmCapabilityMixin, _TerminalBase):
    """Minimal driver: mixin + fakes, no BaseDriver/ROS."""

    use_base_commands = True

    def __init__(self, cfg: ArmKinematicsConfig | None) -> None:
        self._cfg = cfg
        self.applied: list[dict[str, float]] = []
        self.failures: list[tuple[str | None, str]] = []
        self.home_pending = False
        self._joints = {"joint1": 0.5, "joint2": 0.0}
        self._arm_kin_obj = None
        self._arm_cfg_cached = None

    # seams
    def arm_config(self):
        return self._cfg

    def _apply_joint_targets(self, positions):
        self.applied.append(dict(positions))

    def _current_joint_positions(self):
        return dict(self._joints)

    def _on_arm_command_failed(self, command, reason):
        self.failures.append((command, reason))

    # override kinematics factory to avoid pinocchio
    def _build_arm_kinematics(self, config):
        return _FakeKin(config)

    # capture home flag (mixin sets self._home_pending)
    @property
    def _home_pending(self):
        return self.home_pending

    @_home_pending.setter
    def _home_pending(self, value):
        self.home_pending = value


def _registered_command_names(driver: _Driver) -> set[str]:
    iface = DriverInterfaceRegistry()
    # driver.define_interface resolves to ArmCapabilityMixin.define_interface,
    # whose super() call reaches _TerminalBase's no-op.
    driver.define_interface(iface)
    table = iface.command_dispatch_table(DriverOperationMode.TELEOP_LOCAL)
    return set(table.keys())


def test_disabled_when_use_base_commands_false() -> None:
    class Off(_Driver):
        use_base_commands = False

    driver = Off(_full_cfg())
    assert _registered_command_names(driver) == set()


def test_full_config_registers_all_commands() -> None:
    driver = _Driver(_full_cfg())
    names = _registered_command_names(driver)
    assert {
        "ee_move", "ee_rotate_left", "ee_rotate_right", "ee_rotate_up",
        "ee_rotate_down", "base_rotate_left", "base_rotate_right",
        "grip", "release", "home",
    } == names


def test_partial_config_gates_commands() -> None:
    cfg = ArmKinematicsConfig(urdf_path="", ee_frame="", arm_joints=())  # nothing but home
    driver = _Driver(cfg)
    names = _registered_command_names(driver)
    assert names == {"home"}


def test_grip_maps_force_and_expands_mimics() -> None:
    driver = _Driver(_full_cfg())
    driver._on_arm_command({"command": "grip", "data": {"force": 0.5}})
    # target = open + force*(closed-open) = 0 + 0.5*0.04 = 0.02; mimic = -1 * 0.02
    assert driver.applied[-1] == {"jg": 0.02, "jg_mimic": -0.02}


def test_release_opens_gripper() -> None:
    driver = _Driver(_full_cfg())
    driver._on_arm_command({"command": "release", "data": {}})
    assert driver.applied[-1] == {"jg": 0.0, "jg_mimic": -0.0}


def test_base_rotate_nudges_base_joint_within_limits() -> None:
    driver = _Driver(_full_cfg())
    # current joint1 = 0.5; +1.0 rad → clamp to upper limit 1.0
    driver._on_arm_command({"command": "base_rotate_left", "data": {"angle": 1.0}})
    assert driver.applied[-1] == {"joint1": 1.0}


def test_ee_move_solves_and_applies() -> None:
    driver = _Driver(_full_cfg())
    driver._on_arm_command({"command": "ee_move", "data": {"forward": 0.1}})
    assert driver.applied[-1] == {"joint1": 0.25, "joint2": 0.25}


def test_home_sets_pending_flag() -> None:
    driver = _Driver(_full_cfg())
    driver._on_arm_command({"command": "home", "data": {}})
    assert driver.home_pending is True


def test_home_command_calls_request_home_when_available() -> None:
    from cyberwave.driver.control.arm import ArmCapabilityMixin

    class _D(ArmCapabilityMixin):
        def __init__(self) -> None:
            self.homed = False

        def request_home(self) -> None:
            self.homed = True

    d = _D()
    d._on_arm_command({"command": "home", "data": {}})
    assert d.homed is True


def test_gripper_hold_defaults_to_open() -> None:
    d = _Driver(_full_cfg())
    assert d.gripper_hold() == {"jg": 0.0}  # open, commanded name only (no mimic)


def test_latch_and_hold_closed() -> None:
    d = _Driver(_full_cfg())
    d.latch_gripper({"jg": 0.04})           # commanded closed
    assert d.gripper_hold() == {"jg": 0.04}


def test_latch_ignores_absent_gripper() -> None:
    d = _Driver(_full_cfg())
    d.latch_gripper({"joint1": 0.5})        # no gripper joint present
    assert d.gripper_hold() == {"jg": 0.0}  # stays default open


def test_binarize_snaps_gripper_and_present_mimic() -> None:
    d = _Driver(_full_cfg())
    out = d.binarize_gripper({"joint1": 0.5, "jg": 0.03, "jg_mimic": -0.03})
    assert out["joint1"] == 0.5             # untouched
    assert out["jg"] == 0.04                # 0.03 nearer closed(0.04) than open(0.0)
    assert out["jg_mimic"] == -0.04         # factor -1.0 * snapped


def test_binarize_leaves_absent_mimic_absent() -> None:
    d = _Driver(_full_cfg())
    out = d.binarize_gripper({"jg": 0.001})  # mimic not present
    assert out["jg"] == 0.0                  # nearer open
    assert "jg_mimic" not in out


def test_ik_failure_reports_without_raising() -> None:
    class NoIK(_Driver):
        def _build_arm_kinematics(self, config):
            kin = _FakeKin(config)
            kin.ik = lambda start, target: None  # type: ignore[assignment]
            return kin

    driver = NoIK(_full_cfg())
    driver._on_arm_command({"command": "ee_move", "data": {"forward": 0.1}})
    assert driver.applied == []
    assert driver.failures and driver.failures[-1][0] == "ee_move"


def test_on_arm_command_failed_emits_throttled_alert():
    from cyberwave.driver.control.arm import ArmCapabilityMixin

    class _D(ArmCapabilityMixin):
        def __init__(self):
            self.alerts = []
        def create_twin_alert(self, name, **kwargs):
            self.alerts.append((name, kwargs.get("alert_type")))

    d = _D()
    d._on_arm_command_failed("ee_move", "out of reach")
    d._on_arm_command_failed("ee_move", "out of reach")  # throttled → no 2nd alert
    assert d.alerts == [("ik_failed", "ik_failed")]
