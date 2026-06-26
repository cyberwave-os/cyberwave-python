"""Tests for ROS message serialization."""

from __future__ import annotations

from dataclasses import dataclass

from cyberwave.driver.ros2.message_payload import (
    joint_positions_from_transport_payload,
    ros_joint_state_to_transport_payload,
    ros_message_to_transport_payload,
)


@dataclass
class _Stamp:
    sec: int
    nanosec: int


@dataclass
class _Header:
    stamp: _Stamp


@dataclass
class _JointStateMsg:
    header: _Header
    name: list[str]
    position: list[float]
    velocity: list[float]
    effort: list[float]


@dataclass
class _FakeRosMsg:
    data: str = "hello"
    count: int = 3


def test_ros_message_to_transport_payload_dataclass() -> None:
    payload = ros_message_to_transport_payload(_FakeRosMsg())
    assert payload["data"] == "hello"
    assert payload["count"] == 3
    assert payload["source_type"] == "edge"
    assert "timestamp" in payload


def test_ros_joint_state_to_transport_payload_ursim_shape() -> None:
    msg = _JointStateMsg(
        header=_Header(stamp=_Stamp(sec=1780065302, nanosec=945291876)),
        name=[
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "shoulder_pan_joint",
        ],
        position=[
            -0.13753194505642785,
            -0.412595835170098,
            -0.412595835170392,
            0.4189795999984096,
            -0.008142666666812737,
            -0.8251916703401871,
        ],
        velocity=[0.0, 0.0, 0.0, -0.0, 0.0, 0.0],
        effort=[
            -5.874156752602802,
            -1.7486088662265717,
            -0.2097256993350403,
            0.05416192771166935,
            0.0,
            5.4113099533218586e-17,
        ],
    )
    payload = ros_joint_state_to_transport_payload(msg)
    assert payload is not None
    assert payload["source_type"] == "edge"
    assert payload["timestamp"] == 1780065302.945291876
    assert payload["elbow_joint"] == -0.412595835170098
    assert payload["velocities"]["wrist_2_joint"] == -0.0
    assert payload["efforts"]["wrist_1_joint"] == -0.2097256993350403
    assert "positions" not in payload
    assert "name" not in payload
    assert "header" not in payload


def test_ros_joint_state_to_transport_payload_aggregated() -> None:
    msg = _JointStateMsg(
        header=_Header(stamp=_Stamp(sec=1, nanosec=0)),
        name=["j1"],
        position=[0.5],
        velocity=[0.1],
        effort=[0.2],
    )
    payload = ros_joint_state_to_transport_payload(msg, aggregated=True)
    assert payload is not None
    assert payload["positions"] == {"j1": 0.5}
    assert payload["velocities"] == {"j1": 0.1}
    assert payload["efforts"] == {"j1": 0.2}
    assert "j1" not in payload


def test_joint_positions_from_transport_payload_flat_and_aggregated() -> None:
    flat = {
        "source_type": "edge",
        "j1": 0.1,
        "j2": 0.2,
        "velocities": {"j1": 0.0},
        "timestamp": 1.0,
    }
    assert joint_positions_from_transport_payload(flat) == {"j1": 0.1, "j2": 0.2}
    assert joint_positions_from_transport_payload(
        {"positions": {"j1": 0.5}, "source_type": "edge"}
    ) == {"j1": 0.5}


def test_ros_joint_state_to_transport_payload_empty_names() -> None:
    assert (
        ros_joint_state_to_transport_payload(
            _JointStateMsg(
                header=_Header(stamp=_Stamp(sec=0, nanosec=0)),
                name=[],
                position=[],
                velocity=[],
                effort=[],
            )
        )
        is None
    )
