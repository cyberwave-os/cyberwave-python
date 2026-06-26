"""create_twin factory and JointTwin selection."""

from types import SimpleNamespace

from cyberwave.twin.classes import (
    GripperJointTwin,
    JointTwin,
    LocomoteJointCameraTwin,
    LocomoteJointTwin,
)
from cyberwave.twin.factory import _is_joint_manipulator, _select_twin_class, create_twin


def test_so101_selects_gripper_joint_twin() -> None:
    caps = {
        "has_joints": True,
        "can_locomote": False,
        "can_grip": True,
        "sensors": [],
    }
    assert _is_joint_manipulator(caps)
    assert _select_twin_class(caps) is GripperJointTwin


def test_legged_robot_without_sensors_selects_locomote_joint_twin() -> None:
    caps = {"has_joints": True, "can_locomote": True, "can_fly": False, "can_grip": False}
    assert not _is_joint_manipulator(caps)
    assert _select_twin_class(caps) is LocomoteJointTwin


def test_go2_selects_locomote_joint_camera_twin() -> None:
    """Go2 has has_joints + can_locomote + RGB/LIDAR sensors."""
    caps = {
        "has_joints": True,
        "can_locomote": True,
        "can_fly": False,
        "can_grip": False,
        "sensors": [{"type": "rgb"}, {"type": "lidar_4d"}],
    }
    assert not _is_joint_manipulator(caps)
    assert _select_twin_class(caps) is LocomoteJointCameraTwin


def test_create_twin_returns_subclass() -> None:
    client = SimpleNamespace()
    twin = create_twin(
        client,
        SimpleNamespace(uuid="t", capabilities={"has_joints": True, "can_locomote": False}),
        registry_id="the-robot-studio/so101",
    )
    assert isinstance(twin, (JointTwin, GripperJointTwin))
