"""ROS 2 topic spec for Cyberwave driver registry ``from_ros`` publishers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Ros2TopicSpec:
    """ROS subscription source for a Cyber ``add_publisher`` entry.

    Message type is resolved at wire time via the ROS graph unless
    :attr:`msg_type` is set (tests or explicit override).
    """

    topic: str
    qos_depth: int = 10
    qos_profile: Any | None = None
    discovery_timeout_s: float = 5.0
    discovery_poll_interval_s: float = 0.25
    msg_type: type | None = None
