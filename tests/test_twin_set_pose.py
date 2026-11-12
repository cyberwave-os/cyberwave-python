"""``twin.set_pose`` / ``twin.move`` / ``twin.pose.set`` on an arm send ``set_pose``."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cyberwave.data.state_representation import (
    CartesianPose,
    Quaterniond,
    SpatialState,
    Vector3d,
)
from cyberwave.navigation import NavigationPlan
from cyberwave.twin.classes import JointTwin, LocomoteJointTwin


def _client():
    return SimpleNamespace(
        mqtt=MagicMock(),
        assets=MagicMock(),
        config=SimpleNamespace(
            runtime_mode="live", topic_prefix="", source_type="tele"
        ),
        twins=SimpleNamespace(api=None),
    )


@pytest.fixture
def arm():
    twin = JointTwin(_client(), SimpleNamespace(uuid="arm", name="arm", asset_uuid="a"))
    with patch.object(twin, "publish_command") as publish:
        yield twin, publish


def _sent(publish):
    (command, data), kwargs = publish.call_args
    assert command == "set_pose"
    return data, kwargs


def test_only_the_pose_is_required_and_defaults_stay_off_the_wire(arm):
    twin, publish = arm
    twin.set_pose([0.3, 0.0, 0.2, 0.0, 0.0, 0.0])
    assert _sent(publish) == (
        {"pose": [0.3, 0.0, 0.2, 0.0, 0.0, 0.0]},
        {"source_type": None},
    )


def test_a_cartesian_pose_is_sent_with_its_frame(arm):
    """``twin.pose.get()`` comes back as a CartesianPose and goes straight out."""
    twin, publish = arm
    pose = CartesianPose(
        SpatialState("base_link"),
        Vector3d(0.3, 0.0, 0.2),
        Quaterniond(0.0, 0.0, 0.0, 1.0),
    )
    twin.set_pose(pose)
    assert _sent(publish)[0] == {
        "pose": {
            "position": {"x": 0.3, "y": 0.0, "z": 0.2},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "reference_frame": "base_link",
        }
    }


def test_a_position_and_a_rotation_array_are_a_pose(arm):
    twin, publish = arm
    twin.set_pose(([0.3, 0.0, 0.2], [0.0, 0.0, 1.57]))
    assert _sent(publish)[0] == {"pose": [0.3, 0.0, 0.2, 0.0, 0.0, 1.57]}
    twin.set_pose(((0.3, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0)))
    assert _sent(publish)[0] == {"pose": [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]}


@pytest.mark.parametrize(
    "pose",
    [
        [0.3, 0.0, 0.2],
        {"position": [0.3, 0.0, 0.2]},
        ([0.3, 0.0, 0.2], [0.0, 0.0]),
    ],
)
def test_a_pose_without_a_rotation_is_refused(arm, pose):
    twin, publish = arm
    with pytest.raises(ValueError):
        twin.set_pose(pose)
    publish.assert_not_called()


def test_a_cartesian_pose_has_no_degrees(arm):
    """Angles are radians or a quaternion; a degrees switch would be ignored."""
    twin, publish = arm
    with pytest.raises(TypeError, match="no degrees"):
        twin.set_pose([0.3, 0.0, 0.2, 0.0, 90.0, 0.0], degrees=True)
    publish.assert_not_called()


def test_options_are_sent_only_when_they_differ_from_the_default(arm):
    twin, publish = arm
    twin.set_pose(
        [0.05, 0, 0, 0, 0, 0],
        mode="relative",
        motion="lin",
        reference_frame="tool",
        backend="pinocchio",
        expected_joints={"joint1": 0.1},
        scene_revision="r1",
        plan_only=True,
    )
    data, _ = _sent(publish)
    assert data == {
        "pose": [0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
        "mode": "relative",
        "motion": "lin",
        "reference_frame": "tool",
        "backend": "pinocchio",
        "expected_joints": {"joint1": 0.1},
        "scene_revision": "r1",
        "plan_only": True,
    }


def test_move_and_pose_set_are_the_same_command(arm):
    twin, publish = arm
    twin.move([0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], motion="lin")
    assert _sent(publish)[0] == {
        "pose": [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0],
        "motion": "lin",
    }
    twin.pose.set([0.3, 0.0, 0.2, 0.0, 0.0, 0.0], frame_id="link6")
    assert _sent(publish)[0] == {
        "pose": [0.3, 0.0, 0.2, 0.0, 0.0, 0.0],
        "reference_frame": "link6",
    }


def test_a_navigation_waypoint_is_a_pose(arm):
    twin, publish = arm
    waypoint = NavigationPlan().waypoint(0.3, 0.0, 0.2, yaw=0.0).waypoints[0]
    twin.set_pose({**waypoint, "reference_frame": "base_link"})
    pose = _sent(publish)[0]["pose"]
    assert pose == {
        "position": {"x": 0.3, "y": 0.0, "z": 0.2},
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        "reference_frame": "base_link",
    }


def test_an_ambiguous_orientation_list_is_refused(arm):
    twin, publish = arm
    with pytest.raises(ValueError, match="orientation must be a named"):
        twin.set_pose({"position": [0.3, 0, 0.2], "orientation": [0, 0, 0, 1]})
    with pytest.raises(ValueError, match="pose must be"):
        twin.set_pose([0.3, 0.0, 0.2, 0.0])
    publish.assert_not_called()


def test_joint_positions_still_set_the_joints_with_a_deprecation(arm):
    twin, publish = arm
    with patch.object(twin, "set_joints") as set_joints:
        with pytest.warns(DeprecationWarning, match="use set_joints"):
            twin.set_pose({"joint1": 0.5}, degrees=True)
        # A gantry's joints may be called x / y / z: still joints, not a pose.
        with pytest.warns(DeprecationWarning, match="use set_joints"):
            twin.set_pose({"x": 0.1, "y": 0.2, "z": 0.3})
    assert set_joints.call_args_list[0].kwargs["degrees"] is True
    assert set_joints.call_args_list[1].args[0] == {"x": 0.1, "y": 0.2, "z": 0.3}
    publish.assert_not_called()


def test_a_mobile_manipulator_keeps_moving_its_base_with_pose_set():
    assert LocomoteJointTwin._pose_set_moves_the_tool is False
    assert JointTwin._pose_set_moves_the_tool is True
