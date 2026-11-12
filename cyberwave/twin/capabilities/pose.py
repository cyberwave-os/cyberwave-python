"""Versioned MQTT pose read/write handle."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Literal

from ...consumers.callback_hub import CallbackHub, StateSubscription
from ...consumers.mqtt_live_view import PoseView
from ...data.state_representation import (
    CartesianPose,
    cartesian_pose_from_direct_payload,
    cartesian_pose_from_position_payload,
    cartesian_pose_from_rotation_payload,
    decode_kinematics_protobuf_stub,
    merge_cartesian_pose,
)
from ...exceptions import TwinStateUnavailableError
from ...manifest.driver_config import (
    TWIN_KINEMATICS_TOPIC_SLUG,
    TWIN_POSE_TOPIC_SLUG,
    TWIN_POSITION_TOPIC_SLUG,
    TWIN_ROTATION_TOPIC_SLUG,
)
from ...mqtt.state import attach_topic_listener, ensure_mqtt_connected
from ...schema import Quaternion
from .._helpers import _default_control_source_type
from ..runtime_state import (
    active_runtime_mode,
    new_runtime_ready_events,
    runtime_mode_from_mqtt_source_type,
)

if TYPE_CHECKING:
    from ..base import Twin


logger = logging.getLogger(__name__)

PoseVersion = Literal["v0", "v1"]
_V0: PoseVersion = "v0"
_V1: PoseVersion = "v1"


def _validate_version(version: str) -> PoseVersion:
    if version not in {_V0, _V1}:
        raise ValueError(f"Unsupported pose version {version!r}; use 'v0' or 'v1'")
    return version  # type: ignore[return-value]


def _pose_values(pose: Sequence[float], *, length: int, label: str) -> list[float]:
    if isinstance(pose, (str, bytes)) or len(pose) != length:
        raise ValueError(f"{label} must be a {length}-element array")
    try:
        values = [float(value) for value in pose]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only numeric values") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must contain only finite values")
    return values


class PoseHandle:
    def __init__(self, twin: Twin) -> None:
        self._twin = twin
        self._curr_pose_by_key: dict[tuple[PoseVersion, str], CartesianPose] = {}
        self._pose_attached_topics: set[str] = set()
        self._pose_listeners_attached: set[PoseVersion] = set()
        self._pose_ready_by_version = {
            _V0: new_runtime_ready_events(),
            _V1: new_runtime_ready_events(),
        }
        self._pose_lock = threading.Lock()
        self._pose_hub_by_version = {
            _V0: CallbackHub(label="pose-v0"),
            _V1: CallbackHub(label="pose-v1"),
        }
        self._pose_view_by_key: dict[tuple[PoseVersion, str], PoseView] = {}

    def _topic_prefix(self) -> str:
        config = getattr(getattr(self._twin, "client", None), "config", None)
        return getattr(config, "topic_prefix", None) or ""

    def _active_runtime_mode(self) -> str:
        return active_runtime_mode(self._twin.client)

    def _safe_ensure_pose_listeners(self, version: PoseVersion) -> None:
        try:
            self._ensure_pose_listeners(version)
        except TwinStateUnavailableError:
            pass

    def _snapshot_for_mode(
        self, version: PoseVersion, mode: str
    ) -> "CartesianPose | None":
        with self._pose_lock:
            return self._curr_pose_by_key.get((version, mode))

    def _merge_pose(
        self,
        update: CartesianPose,
        *,
        version: PoseVersion,
        runtime_mode: str,
    ) -> None:
        with self._pose_lock:
            key = (version, runtime_mode)
            base = self._curr_pose_by_key.get(key)
            merged = merge_cartesian_pose(base, update)
            self._curr_pose_by_key[key] = merged
            self._pose_ready_by_version[version][runtime_mode].set()
        self._pose_hub_by_version[version].notify(lambda p=merged: p)

    def _on_position_payload(self, payload: dict[str, Any]) -> None:
        mode = runtime_mode_from_mqtt_source_type(
            payload.get("source_type"),
            client=self._twin.client,
        )
        self._merge_pose(
            cartesian_pose_from_position_payload(payload),
            version=_V0,
            runtime_mode=mode,
        )

    def _on_rotation_payload(self, payload: dict[str, Any]) -> None:
        mode = runtime_mode_from_mqtt_source_type(
            payload.get("source_type"),
            client=self._twin.client,
        )
        self._merge_pose(
            cartesian_pose_from_rotation_payload(payload),
            version=_V0,
            runtime_mode=mode,
        )

    def _on_kinematics_payload(self, payload: dict[str, Any]) -> None:
        mode = runtime_mode_from_mqtt_source_type(
            payload.get("source_type"),
            client=self._twin.client,
        )
        msg = decode_kinematics_protobuf_stub(payload)
        if msg.cartesian_pose is not None:
            self._merge_pose(msg.cartesian_pose, version=_V0, runtime_mode=mode)

    def _on_direct_pose_payload(self, payload: dict[str, Any]) -> None:
        mode = runtime_mode_from_mqtt_source_type(
            payload.get("source_type"),
            client=self._twin.client,
        )
        try:
            pose = cartesian_pose_from_direct_payload(payload)
        except ValueError:
            logger.warning("Ignoring malformed direct pose feedback")
            return
        self._merge_pose(pose, version=_V1, runtime_mode=mode)

    def _ensure_pose_listeners(self, version: PoseVersion) -> None:
        if version in self._pose_listeners_attached:
            return
        prefix = self._topic_prefix()
        if version == _V0:
            topics = (
                (TWIN_POSITION_TOPIC_SLUG, self._on_position_payload),
                (TWIN_ROTATION_TOPIC_SLUG, self._on_rotation_payload),
                # CartesianStateMixin publishes the measured TCP pose here.
                (TWIN_KINEMATICS_TOPIC_SLUG, self._on_kinematics_payload),
            )
        else:
            topics = ((TWIN_POSE_TOPIC_SLUG, self._on_direct_pose_payload),)
        for slug, callback in topics:
            attach_topic_listener(
                self._twin,
                topic=f"{prefix}{slug.format(twin_uuid=self._twin.uuid)}",
                on_payload=callback,
                attached_topics=self._pose_attached_topics,
            )
        self._pose_listeners_attached.add(version)

    def _await_initial_pose(self, *, version: PoseVersion, timeout: float) -> None:
        """Wait for the first versioned pose MQTT for the active mode."""
        mode = self._active_runtime_mode()
        if self._pose_ready_by_version[version][mode].is_set():
            return
        with self._pose_lock:
            if (version, mode) in self._curr_pose_by_key:
                return
        self._pose_ready_by_version[version][mode].wait(timeout=timeout)

    def _live_view_for_mode(self, version: PoseVersion, mode: str) -> PoseView:
        key = (version, mode)
        view = self._pose_view_by_key.get(key)
        if view is None:
            view = PoseView(
                self._pose_hub_by_version[version],
                lambda v=version, m=mode: self._snapshot_for_mode(v, m),
            )
            self._pose_view_by_key[key] = view
        return view

    def get(self, *, timeout: float = 3.0, version: str = _V0) -> PoseView:
        """Return a live pose view for the active runtime mode.

        ``version="v0"`` (default) reads the legacy ``/position`` and
        ``/rotation`` streams and the measured TCP pose on
        ``state/kinematics``. ``version="v1"`` reads only the canonical
        ``/pose`` stream carrying ``pose_xyzw``. The result refreshes in place;
        before feedback it reads ``None`` rather than a fabricated zero pose.
        """
        pose_version = _validate_version(version)
        mode = self._active_runtime_mode()
        self._safe_ensure_pose_listeners(pose_version)
        if (pose_version, mode) not in self._curr_pose_by_key:
            self._await_initial_pose(version=pose_version, timeout=timeout)
        return self._live_view_for_mode(pose_version, mode)

    def on_update(
        self,
        callback: "Callable[[CartesianPose | None], None]",
        *,
        version: str = _V0,
    ) -> StateSubscription:
        """Register *callback* for the selected versioned pose stream."""
        pose_version = _validate_version(version)
        self._safe_ensure_pose_listeners(pose_version)
        return self._pose_hub_by_version[pose_version].subscribe(callback)

    def frame_id(self, *, timeout: float = 3.0, version: str = _V0) -> str | None:
        return self.get(timeout=timeout, version=version).frame_id()

    @property
    def translation(self) -> dict[str, float] | None:
        return self.get().translation_dict()

    @property
    def orientation(self) -> dict[str, float] | None:
        return self.get().orientation_dict()

    def set(
        self,
        pose: Any,
        *,
        version: str = _V0,
        frame_id: str | None = None,
        source_type: str | None = None,
        **options: Any,
    ) -> None:
        """Set the twin's pose.

        On an arm this moves the tool: it is :meth:`JointTwin.set_pose
        <cyberwave.twin.mixins.JointsCapableMixin.set_pose>`, with ``frame_id``
        as the ``reference_frame`` and *options* (``mode``, ``motion``,
        ``backend``, ...) passed through; *pose* takes every form that method
        takes, a :class:`CartesianPose` included, and is always a position
        plus a rotation.

        Elsewhere it publishes a complete pose update in the requested MQTT
        contract version. v0 accepts ``[x_m, y_m, z_m, roll_rad, pitch_rad,
        yaw_rad]`` and emits legacy ``/position`` plus ``/rotation`` messages.
        v1 accepts ``[x_m, y_m, z_m, qx, qy, qz, qw]`` and emits one ``/pose``
        message; v1 requires ``frame_id`` and a unit XYZW quaternion.
        """
        if getattr(self._twin, "_pose_set_moves_the_tool", False):
            if frame_id is not None:
                options.setdefault("reference_frame", frame_id)
            self._twin.set_pose(pose, source_type=source_type, **options)  # type: ignore[attr-defined]
            return
        if options:
            raise TypeError(f"unexpected arguments: {sorted(options)}")
        pose_version = _validate_version(version)
        timestamp_ms = int(time.time() * 1_000)
        resolved_source = source_type or _default_control_source_type(self._twin.client)
        prefix = self._topic_prefix()

        if pose_version == _V0:
            x, y, z, roll, pitch, yaw = _pose_values(
                pose, length=6, label="v0 pose"
            )
            quaternion = Quaternion.from_rpy(roll, pitch, yaw)
            base_payload: dict[str, Any] = {
                "source_type": resolved_source,
                "timestamp_ms": timestamp_ms,
            }
            if frame_id:
                base_payload["frame_id"] = frame_id
            messages = (
                (
                    TWIN_POSITION_TOPIC_SLUG,
                    {
                        **base_payload,
                        "type": "position",
                        "position": {"x": x, "y": y, "z": z},
                    },
                ),
                (
                    TWIN_ROTATION_TOPIC_SLUG,
                    {
                        **base_payload,
                        "type": "rotation",
                        "rotation": {
                            "x": quaternion.x,
                            "y": quaternion.y,
                            "z": quaternion.z,
                            "w": quaternion.w,
                        },
                    },
                ),
            )
        else:
            values = _pose_values(pose, length=7, label="v1 pose_xyzw")
            quaternion_norm = math.sqrt(sum(value * value for value in values[3:]))
            if not math.isclose(quaternion_norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
                raise ValueError("v1 pose_xyzw quaternion must be unit length")
            if not frame_id:
                raise ValueError("v1 pose updates require frame_id")
            messages = (
                (
                    TWIN_POSE_TOPIC_SLUG,
                    {
                        "source_type": resolved_source,
                        "frame_id": frame_id,
                        "pose_xyzw": values,
                        "timestamp_ms": timestamp_ms,
                    },
                ),
            )

        ensure_mqtt_connected(self._twin)
        mqtt = getattr(self._twin.client, "mqtt", None)
        if mqtt is None:
            raise TwinStateUnavailableError("MQTT client is not available on this twin")
        for slug, payload in messages:
            mqtt.publish(f"{prefix}{slug.format(twin_uuid=self._twin.uuid)}", payload)
