"""GripperHandle payloads."""

from __future__ import annotations

from unittest.mock import MagicMock

from cyberwave.twin.capabilities.gripper import GripperHandle


def _grip_data(**kwargs) -> dict:
    twin = MagicMock()
    GripperHandle(twin).grip(**kwargs)
    return twin._resolve_topic_and_payload.call_args.kwargs["data"]


def test_grip_sends_max_effort_only_when_given() -> None:
    assert _grip_data() == {"force": 1.0}
    assert _grip_data(force=0.7, max_effort_nm=0.8) == {
        "force": 0.7,
        "max_effort_nm": 0.8,
    }


def test_grip_and_release_ask_for_at_least_once_delivery() -> None:
    """Absolute jaw endpoints are safe to replay; the transport carries no list."""
    twin = MagicMock()
    handle = GripperHandle(twin)
    handle.grip()
    assert twin._resolve_topic_and_payload.call_args.kwargs["qos"] == 1
    handle.release()
    assert twin._resolve_topic_and_payload.call_args.kwargs["qos"] == 1
