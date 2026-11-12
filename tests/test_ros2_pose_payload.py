"""PoseStamped -> canonical ``twin/pose`` payload (``pose_xyzw`` + source stamp)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cyberwave.driver.ros2.message_payload import ros_pose_stamped_to_twin_pose_payload


def _pose(*, w: float = 2.0, frame_id: str = "base_link", sec: int = 1_773_000_000):
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id=frame_id, stamp=SimpleNamespace(sec=sec, nanosec=123_000_000)
        ),
        pose=SimpleNamespace(
            position=SimpleNamespace(x=0.1, y=0.2, z=0.3),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=w),
        ),
    )


def test_pose_stamped_maps_to_normalised_xyzw_with_source_stamp() -> None:
    payload = ros_pose_stamped_to_twin_pose_payload(
        _pose(), child_frame_id="gripper_base"
    )
    assert payload == {
        "source_type": "edge",
        "frame_id": "base_link",
        "child_frame_id": "gripper_base",
        "pose_xyzw": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        "timestamp_ms": 1_773_000_000_123,
    }


def test_frame_override_replaces_an_empty_vendor_header_frame() -> None:
    payload = ros_pose_stamped_to_twin_pose_payload(
        _pose(frame_id=""), frame_id="base_link"
    )
    assert payload is not None
    assert payload["frame_id"] == "base_link"
    assert "child_frame_id" not in payload


@pytest.mark.parametrize(
    "message",
    [
        _pose(w=0.0),  # zero quaternion
        _pose(w=float("nan")),
        _pose(frame_id=""),  # no frame at all
        _pose(sec=-1),
        SimpleNamespace(pose=None, header=None),
    ],
)
def test_unusable_samples_are_dropped(message) -> None:
    assert ros_pose_stamped_to_twin_pose_payload(message) is None


def test_empty_child_frame_is_refused_rather_than_published() -> None:
    assert ros_pose_stamped_to_twin_pose_payload(_pose(), child_frame_id="") is None
