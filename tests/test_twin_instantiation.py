"""create_twin factory and JointTwin selection."""

from types import SimpleNamespace

from cyberwave.twin.classes import (
    CameraTwin,
    DepthCameraTwin,
    GripperJointTwin,
    JointCameraTwin,
    JointDepthCameraTwin,
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


def test_stationary_manipulator_exposes_navigation() -> None:
    """An arm reaches Cartesian waypoints over the navigation contract.

    The Move Twin emitter generates ``twin.navigation.goto(...)`` against a
    twin-link frame for stationary manipulators, which the arm driver plans as a
    Pilz ``MoveGroupSequence``. Without the handle the generated worker dies with
    ``AttributeError: 'GripperJointDepthCameraTwin' object has no attribute
    'navigation'`` at dispatch — after the node has already provisioned a
    billable plant.
    """
    caps = {
        "has_joints": True,
        "can_locomote": False,
        "can_grip": True,
        "sensors": [{"type": "rgb"}, {"type": "depth"}],
    }
    cls = _select_twin_class(caps)
    assert hasattr(cls, "navigation"), (
        f"{cls.__name__} has no .navigation; Move Twin codegen for arms cannot dispatch"
    )


def test_navigation_handle_reaches_every_joint_twin_flavour() -> None:
    """Guard the whole manipulator family, not just the one shape that regressed."""
    for cls in (JointTwin, GripperJointTwin):
        assert hasattr(cls, "navigation"), f"{cls.__name__} lost .navigation"


def test_gripperless_arm_with_camera_keeps_joints_and_navigation() -> None:
    """An arm with a wrist camera but no gripper is still a manipulator.

    ``has_depth``/``has_sensors`` are tested ahead of the manipulator check, so
    this shape needs its own class to keep ``.joints`` and ``.navigation``.
    """
    for sensors, expected in (
        ([{"type": "depth"}], JointDepthCameraTwin),
        ([{"type": "rgb"}], JointCameraTwin),
    ):
        caps = {
            "has_joints": True,
            "can_locomote": False,
            "can_grip": False,
            "sensors": sensors,
        }
        assert _is_joint_manipulator(caps)
        assert _select_twin_class(caps) is expected

        twin = create_twin(
            SimpleNamespace(), SimpleNamespace(uuid="t", capabilities=caps)
        )
        # Asserted on an instance, not hasattr(cls, ...): the property getter has
        # to actually build the handle. Twin.__getattr__ reports a failure in
        # there as the same "no attribute" the generated workers died on.
        assert hasattr(twin.navigation, "goto"), f"{expected.__name__} cannot dispatch"
        assert hasattr(twin.joints, "get"), f"{expected.__name__} lost its joints"


def test_camera_twins_without_joints_keep_their_plain_class() -> None:
    """The manipulator branches must not capture non-jointed sensor twins."""
    for sensors, expected in (
        ([{"type": "depth"}], DepthCameraTwin),
        ([{"type": "rgb"}], CameraTwin),
    ):
        caps = {
            "has_joints": False,
            "can_locomote": False,
            "can_grip": False,
            "sensors": sensors,
        }
        assert not _is_joint_manipulator(caps)
        assert _select_twin_class(caps) is expected
