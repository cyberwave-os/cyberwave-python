"""IMU sensor handle — read latest sample from ``cyberwave/twin/{uuid}/imu``."""

from __future__ import annotations

import copy
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from ...consumers.callback_hub import CallbackHub, StateSubscription
from ...consumers.mqtt_live_view import LiveMappingView
from ...manifest.driver_config import resolve_inbound_topics
from ...mqtt.state import attach_topic_listener
from ..simulation_support import SimLevel, simulation_level

if TYPE_CHECKING:
    from ..base import Twin

__all__ = [
    "ImuSensorHandle",
    "IMU_HANDLE_PUBLIC_METHODS",
    "_is_imu_type",
    "normalize_imu_payload",
]

IMU_HANDLE_PUBLIC_METHODS: tuple[str, ...] = ("metadata", "get", "get_sample", "on_update")


def _is_imu_type(sensor_type: str) -> bool:
    return sensor_type.lower() == "imu"


def normalize_imu_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Map legacy alias keys to canonical ``gyro`` / ``accel`` (no duplicates)."""
    out = dict(payload)
    gyro = out.get("gyro")
    if not isinstance(gyro, dict):
        gyro = out.get("angular_velocity")
    accel = out.get("accel")
    if not isinstance(accel, dict):
        accel = out.get("linear_acceleration")
    if isinstance(gyro, dict):
        out["gyro"] = dict(gyro)
    if isinstance(accel, dict):
        out["accel"] = dict(accel)
    out.pop("angular_velocity", None)
    out.pop("linear_acceleration", None)
    return out


class ImuSensorHandle:
    """Per-sensor IMU façade bound to a twin and sensor id."""

    def __init__(self, twin: "Twin", sensor_id: str) -> None:
        self._twin = twin
        self.sensor_id = sensor_id
        self._curr_imu: dict[str, Any] | None = None
        self._imu_attached_topics: set[str] = set()
        self._imu_listeners_attached = False
        self._imu_ready = threading.Event()
        self._imu_lock = threading.Lock()
        self._imu_hub = CallbackHub(label="imu")
        self._imu_view: LiveMappingView | None = None

    def __repr__(self) -> str:
        methods = ", ".join(IMU_HANDLE_PUBLIC_METHODS)
        return f"{type(self).__name__}(sensor_id={self.sensor_id!r}; {methods})"

    def __dir__(self) -> list[str]:
        names = {n for n in object.__dir__(self) if not n.startswith("_")}
        names.update(IMU_HANDLE_PUBLIC_METHODS)
        return sorted(names)

    def _topic_prefix(self) -> str:
        config = getattr(getattr(self._twin, "client", None), "config", None)
        return getattr(config, "topic_prefix", None) or ""

    def _on_imu_payload(self, payload: dict[str, Any]) -> None:
        raw_sid = payload.get("sensor_id")
        if raw_sid is not None and str(raw_sid) != self.sensor_id:
            return
        with self._imu_lock:
            self._curr_imu = normalize_imu_payload(payload)
            self._imu_ready.set()
        self._imu_hub.notify(self._imu_snapshot)

    def _ensure_imu_listeners(self) -> None:
        if self._imu_listeners_attached:
            return
        topics = resolve_inbound_topics(
            "imu",
            self._twin.driver.get_mqtt_schema(),
            twin_uuid=self._twin.uuid,
            topic_prefix=self._topic_prefix(),
        )
        for _, topic in topics:
            attach_topic_listener(
                self._twin,
                topic=topic,
                on_payload=self._on_imu_payload,
                attached_topics=self._imu_attached_topics,
            )
        self._imu_listeners_attached = True

    def metadata(self) -> Dict[str, Any]:
        """Capability entry for this sensor (rate, axes, frame, …)."""
        for entry in self._twin.capabilities.get("sensors", []):
            if not isinstance(entry, dict):
                continue
            entry_id = str(entry.get("id") or entry.get("name") or "")
            if entry_id == self.sensor_id:
                return dict(entry)
        return {}

    def _safe_ensure_imu_listeners(self) -> None:
        from ...exceptions import TwinStateUnavailableError

        try:
            self._ensure_imu_listeners()
        except TwinStateUnavailableError:
            pass

    def _imu_snapshot(self) -> dict[str, Any]:
        with self._imu_lock:
            return copy.deepcopy(self._curr_imu) if self._curr_imu is not None else {}

    @simulation_level(SimLevel.UNSUPPORTED)
    def get(self, *, timeout: float = 3.0) -> LiveMappingView:
        """Return a **live** IMU view (``cyberwave/twin/{uuid}/imu``).

        The view refreshes in place on every inbound sample and is empty (``{}``)
        until the first sample arrives. If empty, waits up to *timeout* for the
        first sample before returning. The same view is returned on later calls.
        """
        self._safe_ensure_imu_listeners()
        if self._curr_imu is None:
            self._imu_ready.wait(timeout=timeout)
        if self._imu_view is None:
            self._imu_view = LiveMappingView(self._imu_hub, self._imu_snapshot)
        return self._imu_view

    def get_sample(self, *, timeout: Optional[float] = None) -> LiveMappingView:
        """Alias for :meth:`get`."""
        return self.get() if timeout is None else self.get(timeout=timeout)

    def on_update(
        self, callback: "Callable[[dict[str, Any]], None]"
    ) -> StateSubscription:
        self._safe_ensure_imu_listeners()
        return self._imu_hub.subscribe(callback)
