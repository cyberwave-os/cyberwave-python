"""Slice I — JSON legacy → state_representation types."""

from __future__ import annotations

from cyberwave.data.state_representation import (
    CartesianPose,
    cartesian_pose_from_position_payload,
    cartesian_pose_from_rotation_payload,
    decode_joint_update_payload,
    decode_kinematics_protobuf_stub,
    merge_cartesian_pose,
    merge_legacy_position_rotation,
    parse_joint_mqtt_payload,
)


def test_decode_json_legacy_position_rotation_merges() -> None:
    pose = merge_legacy_position_rotation(
        {
            "type": "position",
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "source_type": "edge",
        },
        {
            "type": "rotation",
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "source_type": "edge",
        },
    )
    assert pose.position.x == 1.0
    assert pose.position.y == 2.0
    assert pose.orientation.w == 1.0


def test_merge_cartesian_pose_preserves_position_when_rotation_arrives() -> None:
    pos = cartesian_pose_from_position_payload(
        {"position": {"x": 5.0, "y": 6.0, "z": 7.0}}
    )
    rot = cartesian_pose_from_rotation_payload(
        {"rotation": {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.9}}
    )
    merged = merge_cartesian_pose(pos, rot)
    assert merged.position.x == 5.0
    assert merged.orientation.z == 0.3


def test_decode_joint_flat_payload() -> None:
    joints = decode_joint_update_payload(
        {"_1": 0.5, "_2": 1.0, "source_type": "edge"},
        controllable_names=frozenset({"_1", "_2"}),
    )
    assert joints == {"_1": 0.5, "_2": 1.0}


def test_decode_joint_aggregated_payload() -> None:
    joints = decode_joint_update_payload(
        {
            "positions": {"shoulder": 0.25, "elbow": -0.5},
            "source_type": "tele",
        },
        controllable_names=frozenset({"shoulder", "elbow", "fixed"}),
    )
    assert joints["shoulder"] == 0.25
    assert "fixed" not in joints


def test_parse_joint_mqtt_aggregated_with_velocities_and_efforts() -> None:
    batch = parse_joint_mqtt_payload(
        {
            "positions": {"j1": 1.0, "j2": 2.0},
            "velocities": {"j1": 0.1},
            "efforts": {"j2": 0.5},
            "source_type": "edge",
            "timestamp": 1772035622.5,
        },
        controllable_names=frozenset({"j1", "j2"}),
    )
    assert batch.positions == {"j1": 1.0, "j2": 2.0}
    assert batch.velocities == {"j1": 0.1}
    assert batch.efforts == {"j2": 0.5}


def test_parse_joint_mqtt_flat_ignores_vector_metadata_keys() -> None:
    batch = parse_joint_mqtt_payload(
        {
            "_1": 0.5,
            "source_type": "edge",
            "source_subtype": "keyboard",
            "workload_uuid": "wk-1",
            "session_id": "sess-1",
            "timestamp": 1.0,
            "camera_frame_counters": {"cam": 42},
        },
        controllable_names=frozenset({"_1"}),
    )
    assert batch.positions == {"_1": 0.5}
    assert batch.velocities == {}
    assert batch.efforts == {}


def test_parse_joint_mqtt_single_joint_state() -> None:
    batch = parse_joint_mqtt_payload(
        {
            "joint_name": "shoulder",
            "joint_state": {"position": 0.75, "velocity": 0.05, "effort": 0.02},
            "source_type": "edge",
        },
        controllable_names=frozenset({"shoulder"}),
    )
    assert batch.positions == {"shoulder": 0.75}
    assert batch.velocities == {"shoulder": 0.05}
    assert batch.efforts == {"shoulder": 0.02}


def test_joints_listener_applies_single_joint_payload() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from cyberwave.twin.capabilities.joints import JointsHandle
    from cyberwave.twin.classes import JointTwin

    mqtt = MagicMock()
    mqtt.connected = True
    mqtt._subs = {}

    def _subscribe(topic: str, callback: object) -> None:
        mqtt._subs[topic] = callback

    mqtt.subscribe = MagicMock(side_effect=_subscribe)
    client = SimpleNamespace(
        mqtt=mqtt,
        config=SimpleNamespace(topic_prefix=""),
        twins=SimpleNamespace(),
    )
    twin = JointTwin(
        client,
        SimpleNamespace(uuid="arm-1", name="Arm", capabilities={"has_joints": True}),
    )
    with patch(
        "cyberwave.twin.capabilities.joints.controllable_joint_names",
        return_value=["shoulder"],
    ):
        handle = JointsHandle(twin)
        topic = "cyberwave/joint/arm-1/update"
        mqtt._subs[topic](
            {
                "joint_name": "shoulder",
                "joint_state": {"position": 1.25, "velocity": 0.0, "effort": 0.0},
            }
        )
        assert handle.get(timeout=0.0)["shoulder"] == 1.25


def test_decode_kinematics_protobuf_stub_cartesian() -> None:
    msg = decode_kinematics_protobuf_stub(
        {
            "message_type": "cartesian_pose",
            "cartesian_pose": {
                "reference_frame": "world",
                "position": {"x": 1.0, "y": 0.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        }
    )
    assert msg.which() == "cartesian_pose"
    assert msg.cartesian_pose is not None
    assert msg.cartesian_pose.frame_id() == "world"


def test_cartesian_pose_copy_is_independent() -> None:
    pose = merge_legacy_position_rotation(
        {"position": {"x": 1.0, "y": 0.0, "z": 0.0}},
        None,
    )
    copy = pose.copy()
    assert isinstance(copy, CartesianPose)
    assert copy.position.x == pose.position.x
