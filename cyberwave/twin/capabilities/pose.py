"""Pose state handle (MQTT listeners — plane C read, plane B write stub)."""

from __future__ import annotations

import copy
import threading
from typing import TYPE_CHECKING, Any

from ...data.state_representation import (
    CartesianPose,
    cartesian_pose_from_position_payload,
    cartesian_pose_from_rotation_payload,
    merge_cartesian_pose,
)
from ...data.state_representation.geometry.primitives import Quaterniond, Vector3d
from ...data.state_representation.space.spatial_state import SpatialState
from ...manifest.driver_config import resolve_inbound_topics
from ...mqtt.state import attach_topic_listener
from ..runtime_state import (
    active_runtime_mode,
    new_runtime_ready_events,
    runtime_mode_from_mqtt_source_type,
)

if TYPE_CHECKING:
    from ..base import Twin


class PoseHandle:
    def __init__(self, twin: Twin) -> None:
        self._twin = twin
        self._curr_pose_by_mode: dict[str, CartesianPose] = {}
        self._pose_attached_topics: set[str] = set()
        self._pose_listeners_attached = False
        self._pose_ready_by_mode = new_runtime_ready_events()
        self._pose_lock = threading.Lock()

    def _topic_prefix(self) -> str:
        config = getattr(getattr(self._twin, "client", None), "config", None)
        return getattr(config, "topic_prefix", None) or ""

    def _active_runtime_mode(self) -> str:
        return active_runtime_mode(self._twin.client)

    @staticmethod
    def _default_pose() -> CartesianPose:
        return CartesianPose(
            spatial_state=SpatialState(),
            position=Vector3d(),
            orientation=Quaterniond(),
        )

    def _merge_pose(self, update: CartesianPose, *, runtime_mode: str) -> None:
        with self._pose_lock:
            base = self._curr_pose_by_mode.get(runtime_mode)
            self._curr_pose_by_mode[runtime_mode] = merge_cartesian_pose(base, update)
            self._pose_ready_by_mode[runtime_mode].set()

    def _on_position_payload(self, payload: dict[str, Any]) -> None:
        mode = runtime_mode_from_mqtt_source_type(
            payload.get("source_type"),
            client=self._twin.client,
        )
        self._merge_pose(cartesian_pose_from_position_payload(payload), runtime_mode=mode)

    def _on_rotation_payload(self, payload: dict[str, Any]) -> None:
        mode = runtime_mode_from_mqtt_source_type(
            payload.get("source_type"),
            client=self._twin.client,
        )
        self._merge_pose(cartesian_pose_from_rotation_payload(payload), runtime_mode=mode)

    def _ensure_pose_listeners(self) -> None:
        if self._pose_listeners_attached:
            return
        topics = resolve_inbound_topics(
            "pose",
            self._twin.commands.get_schema(),
            twin_uuid=self._twin.uuid,
            topic_prefix=self._topic_prefix(),
        )
        for slug, topic in topics:
            if slug.endswith("/position") or "/position" in slug:
                attach_topic_listener(
                    self._twin,
                    topic=topic,
                    on_payload=self._on_position_payload,
                    attached_topics=self._pose_attached_topics,
                )
            elif slug.endswith("/rotation") or "/rotation" in slug:
                attach_topic_listener(
                    self._twin,
                    topic=topic,
                    on_payload=self._on_rotation_payload,
                    attached_topics=self._pose_attached_topics,
                )
            elif "kinematics" in slug:
                from ...data.state_representation import decode_kinematics_protobuf_stub

                def _on_kinematics(payload: dict[str, Any], *, _slug: str = slug) -> None:
                    del _slug
                    mode = runtime_mode_from_mqtt_source_type(
                        payload.get("source_type"),
                        client=self._twin.client,
                    )
                    msg = decode_kinematics_protobuf_stub(payload)
                    if msg.cartesian_pose is not None:
                        self._merge_pose(msg.cartesian_pose, runtime_mode=mode)

                attach_topic_listener(
                    self._twin,
                    topic=topic,
                    on_payload=_on_kinematics,
                    attached_topics=self._pose_attached_topics,
                )
        self._pose_listeners_attached = True

    def _await_initial_pose(self, *, timeout: float) -> None:
        """Wait for first pose MQTT for the active mode; on timeout use zero pose."""
        mode = self._active_runtime_mode()
        if self._pose_ready_by_mode[mode].is_set():
            return
        with self._pose_lock:
            if mode in self._curr_pose_by_mode:
                return
        if not self._pose_ready_by_mode[mode].wait(timeout=timeout):
            with self._pose_lock:
                self._curr_pose_by_mode[mode] = self._default_pose()
                self._pose_ready_by_mode[mode].set()

    def get(self, *, timeout: float = 3.0) -> CartesianPose:
        """Read canonical pose from MQTT for the active ``config.runtime_mode``.

        If no MQTT pose arrives within *timeout* for that mode, returns a zero
        pose (origin position, identity orientation). Inbound messages are stored
        under ``live`` or ``simulation`` based on MQTT ``source_type``.
        """
        mode = self._active_runtime_mode()
        self._ensure_pose_listeners()
        wait_timeout = timeout if mode not in self._curr_pose_by_mode else 0.0
        self._await_initial_pose(timeout=wait_timeout)
        with self._pose_lock:
            pose = self._curr_pose_by_mode.get(mode)
            if pose is None:
                pose = self._default_pose()
                self._curr_pose_by_mode[mode] = pose
            return copy.deepcopy(pose)

    def frame_id(self, *, timeout: float = 3.0) -> str:
        return self.get(timeout=timeout).frame_id()

    @property
    def translation(self) -> dict[str, float]:
        return self.get().translation_dict()

    @property
    def orientation(self) -> dict[str, float]:
        return self.get().orientation_dict()

    def set(self, **kwargs: Any) -> None:
        """MQTT pose publish (not implemented in PR3).

        Future: publish via ``update_twin_position`` / ``update_twin_rotation`` and
        keep listeners on those topics so per-mode pose caches stay in sync.
        """
        raise NotImplementedError(
            "twin.pose.set() MQTT publish is not implemented yet. "
            "Use edit_position() / edit_rotation() for scene layout, or "
            "locomotion commands for sim/live control."
        )
