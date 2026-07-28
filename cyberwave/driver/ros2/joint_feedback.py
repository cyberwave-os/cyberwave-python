"""Ros2JointFeedbackMixin — ROS joint feedback -> local state -> edge payload.

Generalizes the chain every ROS arm driver hand-rolled (subscribe to a
JointState topic, maintain ``self._joint_states``, forward to the
``joint/update`` edge publisher). Supplies ``_current_joint_positions()`` — the
read seam ArmCapabilityMixin / EECartesianPoseMixin / JointControllerMixin
require — so a driver composing feedback + controller mixins implements no
joint plumbing at all. One-time vendor setup that must run on first feedback
(e.g. priming a CAN bus) goes in ``on_first_joint_feedback()``.

rclpy is imported lazily inside ``register_callbacks`` so this module stays
importable off-robot (tests, manifest export).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from cyberwave.constants import SOURCE_TYPE_EDGE

from .joint_names import JointNameMap

logger = logging.getLogger(__name__)


class Ros2JointFeedbackMixin:
    """Joint feedback subscription + state + edge-payload conversion."""

    use_joint_feedback: bool = False

    def _init_joint_feedback(self) -> None:
        """Initialize feedback state. Call once from ``__init__``."""
        self._joint_states: dict[str, float] = {}
        self._joint_feedback_count = 0
        self._joint_feedback_sub: Any | None = None

    # ── author-supplied seams ────────────────────────────────────────────────

    def joint_feedback_topic(self) -> str:
        """Relative ROS topic providing JointState feedback."""
        return "joint_states"

    def joint_name_map(self) -> JointNameMap | None:
        """ros↔platform renaming + mimic expansion. None => identity."""
        return None

    def on_first_joint_feedback(self) -> None:
        """Hook invoked once, on the first non-empty feedback message."""

    # ── wiring ───────────────────────────────────────────────────────────────

    def register_callbacks(self) -> None:
        parent = getattr(super(), "register_callbacks", None)
        if parent is not None:
            parent()
        if not type(self).use_joint_feedback:
            return
        if self._joint_feedback_sub is not None:
            return
        from rclpy.qos import qos_profile_sensor_data  # noqa: PLC0415 (lazy)

        from .lazy_ros_msgs import joint_state_message_type  # noqa: PLC0415

        topic = self.joint_feedback_topic()
        self._joint_feedback_sub = self.create_subscription(  # type: ignore[attr-defined]
            joint_state_message_type(),
            topic,
            self._on_joint_feedback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(f"Joint feedback subscription: {topic} (JointState)")  # type: ignore[attr-defined]

    def _feedback_positions(self, msg: Any) -> dict[str, float]:
        names, positions = list(msg.name), list(msg.position)
        name_map = self.joint_name_map()
        if name_map is None:
            return {n: float(p) for n, p in zip(names, positions)}
        return name_map.to_platform(names, positions)

    def _on_joint_feedback(self, msg: Any) -> None:
        positions = self._feedback_positions(msg)
        if not positions:
            return
        self._joint_states.update(positions)
        self._joint_feedback_count += 1
        touch = getattr(self, "_touch_edge_health", None)
        if callable(touch):
            touch()
        if self._joint_feedback_count == 1:
            self.get_logger().info(  # type: ignore[attr-defined]
                f"First {self.joint_feedback_topic()} RX ({len(positions)} joints)"
            )
            self.on_first_joint_feedback()

    # ── read seam + edge forwarding ──────────────────────────────────────────

    def _current_joint_positions(self) -> dict[str, float]:
        """Mutated only on the ROS executor thread; read lock-free."""
        return dict(self._joint_states)

    def convert_joints_to_payload(self, msg: Any) -> dict[str, Any] | None:
        """``from_ros`` forwarder seam: JointState -> flat edge payload (or None).
        Applies ``binarize_gripper`` when an arm mixin provides it."""
        positions = self._feedback_positions(msg)
        if not positions:
            return None
        payload: dict[str, Any] = {
            "source_type": SOURCE_TYPE_EDGE,
            **positions,
            "timestamp": time.time(),
        }
        binarize = getattr(self, "binarize_gripper", None)
        return binarize(payload) if callable(binarize) else payload
