"""Discover ROS 2 message classes from the live topic graph."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rclpy.lifecycle import LifecycleNode

logger = logging.getLogger(__name__)


class RosTopicDiscoveryError(RuntimeError):
    """Raised when a ROS topic or message type cannot be resolved."""


def resolve_ros_message_class(
    node: LifecycleNode,
    topic: str,
    *,
    timeout_s: float,
    poll_interval_s: float,
    msg_type: type | None = None,
) -> tuple[type, str]:
    """Return ``(message_class, type_string)`` for *topic* on the ROS graph."""
    if msg_type is not None:
        type_string = _type_string_for_class(msg_type)
        return msg_type, type_string

    deadline = time.monotonic() + max(0.0, timeout_s)
    poll = max(0.05, poll_interval_s)
    msg_type_string: str | None = None
    last_wait_log_at = 0.0

    while time.monotonic() < deadline:
        msg_type_string = _lookup_topic_type(node, topic)
        if msg_type_string:
            logger.info("Resolved ROS topic %s -> %s", topic, msg_type_string)
            break
        now = time.monotonic()
        if now - last_wait_log_at >= 2.0:
            last_wait_log_at = now
            known = _summarize_joint_topics(node)
            logger.info(
                "Waiting for ROS topic %r on the graph (%.1fs left). "
                "Known joint-related topics: %s",
                topic,
                max(0.0, deadline - now),
                known or "(none)",
            )
        time.sleep(poll)

    if not msg_type_string:
        known = _summarize_joint_topics(node)
        raise RosTopicDiscoveryError(
            f"ROS topic {topic!r} not found within {timeout_s}s — is it publishing? "
            f"Known joint-related topics: {known or '(none)'}"
        )

    try:
        from rosidl_runtime_py.utilities import get_message

        msg_class = get_message(msg_type_string)
    except Exception as exc:
        raise RosTopicDiscoveryError(
            f"Failed to load message class for {msg_type_string!r}: {exc}"
        ) from exc

    return msg_class, msg_type_string


def _summarize_joint_topics(node: LifecycleNode) -> list[str]:
    try:
        names_and_types = node.get_topic_names_and_types()
    except Exception:
        return []
    return sorted(
        name
        for name, _types in names_and_types
        if "joint" in name.lower() or "tcp_pose" in name.lower()
    )[:12]


def _lookup_topic_type(node: LifecycleNode, topic: str) -> str | None:
    try:
        names_and_types = node.get_topic_names_and_types()
    except Exception:
        logger.debug("get_topic_names_and_types failed", exc_info=True)
        return None

    for name, types in names_and_types:
        if name != topic:
            continue
        if not types:
            return None
        if len(types) > 1:
            logger.warning(
                "ROS topic %s has multiple types %s; using %s",
                topic,
                types,
                types[0],
            )
        return types[0]
    return None


def _type_string_for_class(msg_type: type) -> str:
    module = getattr(msg_type, "__module__", "") or ""
    name = getattr(msg_type, "__name__", "") or ""
    if module.startswith("std_msgs.msg"):
        return f"std_msgs/msg/{name}"
    if module.startswith("sensor_msgs.msg"):
        return f"sensor_msgs/msg/{name}"
    if module.startswith("geometry_msgs.msg"):
        return f"geometry_msgs/msg/{name}"
    if "." in module:
        pkg = module.split(".")[0]
        return f"{pkg}/msg/{name}"
    return name
