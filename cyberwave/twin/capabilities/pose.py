"""Pose state handle (MQTT-based read; write not yet implemented)."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Callable

from ...consumers.callback_hub import CallbackHub, StateSubscription
from ...consumers.mqtt_live_view import PoseView
from ...data.state_representation import (
    CartesianPose,
    cartesian_pose_from_position_payload,
    cartesian_pose_from_rotation_payload,
    merge_cartesian_pose,
)
from ...exceptions import TwinStateUnavailableError
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
        self._pose_hub = CallbackHub(label="pose")
        self._pose_view_by_mode: dict[str, PoseView] = {}

    def _topic_prefix(self) -> str:
        config = getattr(getattr(self._twin, "client", None), "config", None)
        return getattr(config, "topic_prefix", None) or ""

    def _active_runtime_mode(self) -> str:
        return active_runtime_mode(self._twin.client)

    def _safe_ensure_pose_listeners(self) -> None:
        try:
            self._ensure_pose_listeners()
        except TwinStateUnavailableError:
            pass

    def _snapshot_for_mode(self, mode: str) -> "CartesianPose | None":
        with self._pose_lock:
            return self._curr_pose_by_mode.get(mode)

    def _merge_pose(self, update: CartesianPose, *, runtime_mode: str) -> None:
        with self._pose_lock:
            base = self._curr_pose_by_mode.get(runtime_mode)
            merged = merge_cartesian_pose(base, update)
            self._curr_pose_by_mode[runtime_mode] = merged
            self._pose_ready_by_mode[runtime_mode].set()
        self._pose_hub.notify(lambda p=merged: p)

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
            self._twin.driver.get_mqtt_schema(),
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
        """Wait for the first pose MQTT for the active mode (no fabrication)."""
        mode = self._active_runtime_mode()
        if self._pose_ready_by_mode[mode].is_set():
            return
        with self._pose_lock:
            if mode in self._curr_pose_by_mode:
                return
        self._pose_ready_by_mode[mode].wait(timeout=timeout)

    def _live_view_for_mode(self, mode: str) -> PoseView:
        view = self._pose_view_by_mode.get(mode)
        if view is None:
            view = PoseView(self._pose_hub, lambda m=mode: self._snapshot_for_mode(m))
            self._pose_view_by_mode[mode] = view
        return view

    def get(self, *, timeout: float = 3.0) -> PoseView:
        """Return a **live** :class:`PoseView` for the active ``config.runtime_mode``.

        The returned view refreshes in place on every inbound MQTT pose and never
        goes stale; the same object is returned on later ``get()`` calls for the
        same mode. Before any pose arrives (or with no MQTT transport) the view's
        ``.pose`` reads ``None`` rather than a fabricated zero pose. If the cache
        is empty, waits up to *timeout* for the first message before returning.
        """
        mode = self._active_runtime_mode()
        self._safe_ensure_pose_listeners()
        if mode not in self._curr_pose_by_mode:
            self._await_initial_pose(timeout=timeout)
        return self._live_view_for_mode(mode)

    def on_update(
        self, callback: "Callable[[CartesianPose | None], None]"
    ) -> StateSubscription:
        """Register *callback* to run on every inbound pose (current mode's pose)."""
        self._safe_ensure_pose_listeners()
        return self._pose_hub.subscribe(callback)

    def frame_id(self, *, timeout: float = 3.0) -> str | None:
        return self.get(timeout=timeout).frame_id()

    @property
    def translation(self) -> dict[str, float] | None:
        return self.get().translation_dict()

    @property
    def orientation(self) -> dict[str, float] | None:
        return self.get().orientation_dict()

    def set(self, **kwargs: Any) -> None:
        """Set the twin pose over MQTT — not yet available in this SDK release."""
        raise NotImplementedError(
            "twin.pose.set() is not yet implemented."
        )
