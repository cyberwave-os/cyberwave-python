"""IMU sensor handle — read latest sample from ``cyberwave/twin/{uuid}/imu``."""

from __future__ import annotations

import copy
import threading
from typing import TYPE_CHECKING, Any, Dict, Optional

from ...manifest.driver_config import resolve_inbound_topics
from ...mqtt.state import attach_topic_listener, wait_for_first_message

if TYPE_CHECKING:
    from ..base import Twin

__all__ = ["ImuSensorHandle", "_is_imu_type", "normalize_imu_payload"]


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

    def __repr__(self) -> str:
        from ..namespaces.imu import IMU_HANDLE_PUBLIC_METHODS

        methods = ", ".join(IMU_HANDLE_PUBLIC_METHODS)
        return f"{type(self).__name__}(sensor_id={self.sensor_id!r}; {methods})"

    def __dir__(self) -> list[str]:
        from ..namespaces.imu import IMU_HANDLE_PUBLIC_METHODS

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

    def get(self, *, timeout: float = 3.0) -> dict[str, Any]:
        """Return the latest IMU sample from MQTT (``cyberwave/twin/{uuid}/imu``)."""
        self._ensure_imu_listeners()
        wait_for_first_message(
            self._imu_ready,
            timeout=timeout if self._curr_imu is None else 0.0,
            twin_uuid=self._twin.uuid,
            stream="imu",
        )
        with self._imu_lock:
            if self._curr_imu is None:
                from ...exceptions import TwinStateTimeoutError

                raise TwinStateTimeoutError(
                    f"No MQTT IMU update within {timeout}s for twin {self._twin.uuid}"
                )
            return copy.deepcopy(self._curr_imu)

    def get_sample(self, *, timeout: Optional[float] = None) -> dict[str, Any]:
        """Alias for :meth:`get` (same payload shape)."""
        if timeout is None:
            return self.get()
        return self.get(timeout=timeout)
