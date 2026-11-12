"""CartesianStateMixin: registration, payload shape, decoder compatibility.

The mixin is a thin state broadcaster over the arm's ``ArmController``: it owns
no kinematics of its own and derives the reference frame from config, so the
frame label can never disagree with the frame FK is measured from.
"""

from __future__ import annotations

import numpy as np

from cyberwave.data.state_representation import decode_kinematics_protobuf_stub
from cyberwave.driver import (
    ArmCapabilityMixin,
    ArmKinematicsConfig,
    CartesianStateMixin,
)
from cyberwave.driver.interface.registry import DriverInterfaceRegistry


class _FakeKin:
    def __init__(self, config):
        self.config = config

    def fk(self, positions):
        pose = np.eye(4)
        pose[:3, 3] = [0.1, 0.2, 0.3]
        return pose

    def apply_translation(self, pose, components, *, frame="tool"):
        return np.eye(4)

    def apply_rotation(self, pose, axis, angle, *, frame="tool"):
        return np.eye(4)

    def ik(self, start, target):
        return {n: 0.25 for n in self.config.arm_joints}

    def joint_limits(self):
        return {n: (-1.0, 1.0) for n in self.config.arm_joints}


def _cfg(base_frame: str = "base_link") -> ArmKinematicsConfig:
    return ArmKinematicsConfig(
        urdf_path="/x.urdf",
        ee_frame="tool",
        arm_joints=("joint1", "joint2"),
        base_frame=base_frame,
        tool_axes={"forward": np.array([0.0, 0.0, 1.0])},
        rot_axes={"yaw": np.array([0.0, 0.0, 1.0])},
    )


class _FakeTwin:
    def __init__(self):
        self.published: list[dict] = []

    def publish_kinematics(self, payload):
        self.published.append(payload)


class _TerminalBase:
    def define_interface(self, iface):
        pass


class _Driver(CartesianStateMixin, ArmCapabilityMixin, _TerminalBase):
    use_cartesian_state = True
    use_base_commands = True

    def __init__(self, *, joints=None, twin=True, base_frame="base_link"):
        super().__init__()
        self._cfg = _cfg(base_frame)
        self._joints = joints if joints is not None else {"joint1": 0.0, "joint2": 0.0}
        self._twin = _FakeTwin() if twin else None
        self.build_count = 0
        self.applied: list[dict] = []

    def arm_config(self):
        return self._cfg

    def _current_joint_positions(self):
        return dict(self._joints)

    def _apply_joint_targets(self, positions):
        self.applied.append(dict(positions))

    def _build_arm_kinematics(self, config):
        self.build_count += 1
        return _FakeKin(config)


def _publisher_keys(driver) -> list:
    iface = DriverInterfaceRegistry()
    driver.define_interface(iface)
    return [p.topic.registry_key() for p in iface._publishers]


def test_publisher_is_registered_on_the_kinematics_topic():
    assert ("ns", "state/kinematics") in _publisher_keys(_Driver())


def test_publisher_is_not_registered_when_disabled():
    class _Off(_Driver):
        use_cartesian_state = False

    assert ("ns", "state/kinematics") not in _publisher_keys(_Off())


def test_payload_carries_position_orientation_and_frame():
    payload = _Driver()._cartesian_state_payload()
    assert payload is not None
    assert payload["message_type"] == "cartesian_pose"
    cp = payload["cartesian_pose"]
    assert cp["position"] == {"x": 0.1, "y": 0.2, "z": 0.3}
    assert cp["orientation"] == {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    assert cp["reference_frame"] == "base_link"


def test_reference_frame_comes_from_the_arm_config():
    """Not a driver-asserted string: the frame FK validated against."""
    payload = _Driver(base_frame="world")._cartesian_state_payload()
    assert payload["cartesian_pose"]["reference_frame"] == "world"


def test_payload_round_trips_through_the_platform_decoder():
    msg = decode_kinematics_protobuf_stub(_Driver()._cartesian_state_payload())
    assert msg.cartesian_pose is not None
    assert msg.cartesian_pose.position.y == 0.2
    assert msg.cartesian_pose.spatial_state.reference_frame == "base_link"


def test_no_payload_before_any_joint_feedback():
    assert _Driver(joints={})._cartesian_state_payload() is None


def test_no_payload_when_only_a_non_arm_joint_has_reported():
    """A startup JointState carrying just the gripper must not publish a pose:
    the controller would substitute 0.0 for every arm joint and the result would
    be labelled 'measured'."""
    assert _Driver(joints={"jg": 0.02})._cartesian_state_payload() is None


def test_no_payload_when_one_arm_joint_is_missing():
    assert _Driver(joints={"joint1": 0.3})._cartesian_state_payload() is None


def test_payload_published_once_every_arm_joint_has_reported():
    d = _Driver(joints={"joint1": 0.3, "joint2": 0.4})
    assert d._cartesian_state_payload() is not None


def test_extra_non_arm_joints_do_not_block_publishing():
    d = _Driver(joints={"joint1": 0.3, "joint2": 0.4, "jg": 0.02})
    assert d._cartesian_state_payload() is not None


def test_callback_does_not_publish_from_partial_feedback():
    d = _Driver(joints={"joint1": 0.3})
    d._cartesian_state_publisher_callback()
    assert d._twin.published == []


def test_state_and_command_paths_share_one_kinematics_model():
    """Regression: the publisher used to build a second pinocchio model."""
    d = _Driver()
    d._on_arm_command({"command": "ee_move", "data": {"forward": 0.05}})
    d._cartesian_state_payload()
    assert d.applied, "command path did not run"
    assert d.build_count == 1


def test_callback_returns_the_payload_for_the_registry_to_publish():
    # Not twin.publish_kinematics: that reconnects to MQTT synchronously and
    # blocked the driver loop for seconds per tick during a broker outage.
    d = _Driver()
    payload = d._cartesian_state_publisher_callback()
    assert payload is not None
    assert payload["message_type"] == "cartesian_pose"
    assert d._twin.published == []


def test_callback_is_a_noop_without_a_twin():
    _Driver(twin=False)._cartesian_state_publisher_callback()  # must not raise


def test_no_payload_without_a_composed_controller():
    """DualArmMixin has no `_arm`; the single-pose payload cannot represent it."""

    class _NoArm(CartesianStateMixin, _TerminalBase):
        use_cartesian_state = True

        def _current_joint_positions(self):
            return {"joint1": 0.0}

    assert _NoArm()._cartesian_state_payload() is None
