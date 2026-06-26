"""Lazy ROS message/service imports for edge drivers (keeps driver sources ROS-msg-free)."""

from __future__ import annotations

from typing import Any


def joint_state_message_type() -> type[Any]:
    from sensor_msgs.msg import JointState

    return JointState


def bool_message_type() -> type[Any]:
    from std_msgs.msg import Bool

    return Bool
