"""GPS sensor handle — read latest fix from ``cyberwave/twin/{uuid}/gps``."""

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
    "GpsSensorHandle",
    "GPS_HANDLE_PUBLIC_METHODS",
    "_is_gps_type",
    "normalize_gps_payload",
]

GPS_HANDLE_PUBLIC_METHODS: tuple[str, ...] = ("metadata", "get_fix", "get", "on_update")


def _is_gps_type(sensor_type: str) -> bool:
    return sensor_type.lower() == "gps"


def normalize_gps_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize a GNSS fix payload (currently a shallow copy)."""
    return dict(payload)


class GpsSensorHandle:
    """Per-sensor GPS façade bound to a twin and sensor id."""

    def __init__(self, twin: "Twin", sensor_id: str) -> None:
        self._twin = twin
        self.sensor_id = sensor_id
        self._curr_gps: dict[str, Any] | None = None
        self._gps_attached_topics: set[str] = set()
        self._gps_listeners_attached = False
        self._gps_ready = threading.Event()
        self._gps_lock = threading.Lock()
        self._gps_hub = CallbackHub(label="gps")
        self._gps_view: LiveMappingView | None = None

    def __repr__(self) -> str:
        methods = ", ".join(GPS_HANDLE_PUBLIC_METHODS)
        return f"{type(self).__name__}(sensor_id={self.sensor_id!r}; {methods})"

    def __dir__(self) -> list[str]:
        names = {n for n in object.__dir__(self) if not n.startswith("_")}
        names.update(GPS_HANDLE_PUBLIC_METHODS)
        return sorted(names)

    def _topic_prefix(self) -> str:
        config = getattr(getattr(self._twin, "client", None), "config", None)
        return getattr(config, "topic_prefix", None) or ""

    def metadata(self) -> Dict[str, Any]:
        """Capability entry for this sensor (rate, frame, accuracy, …)."""
        for entry in self._twin.capabilities.get("sensors", []):
            if not isinstance(entry, dict):
                continue
            entry_id = str(entry.get("id") or entry.get("name") or "")
            if entry_id == self.sensor_id:
                return dict(entry)
        return {}

    def _on_gps_payload(self, payload: dict[str, Any]) -> None:
        raw_sid = payload.get("sensor_id")
        if raw_sid is not None and str(raw_sid) != self.sensor_id:
            return
        with self._gps_lock:
            self._curr_gps = normalize_gps_payload(payload)
            self._gps_ready.set()
        self._gps_hub.notify(self._gps_snapshot)

    def _ensure_gps_listeners(self) -> None:
        if self._gps_listeners_attached:
            return
        topics = resolve_inbound_topics(
            "gps",
            self._twin.driver.get_mqtt_schema(),
            twin_uuid=self._twin.uuid,
            topic_prefix=self._topic_prefix(),
        )
        for _, topic in topics:
            attach_topic_listener(
                self._twin,
                topic=topic,
                on_payload=self._on_gps_payload,
                attached_topics=self._gps_attached_topics,
            )
        self._gps_listeners_attached = True

    def _safe_ensure_gps_listeners(self) -> None:
        from ...exceptions import TwinStateUnavailableError

        try:
            self._ensure_gps_listeners()
        except TwinStateUnavailableError:
            pass

    def _gps_snapshot(self) -> dict[str, Any]:
        with self._gps_lock:
            return copy.deepcopy(self._curr_gps) if self._curr_gps is not None else {}

    @simulation_level(SimLevel.UNSUPPORTED)
    def get_fix(self, *, timeout: float = 3.0) -> LiveMappingView:
        """Return a **live** GNSS fix view (``cyberwave/twin/{uuid}/gps``).

        Refreshes in place on every inbound fix; empty (``{}``) until the first
        fix arrives. If empty, waits up to *timeout* for the first fix. The same
        view is returned on later calls.
        """
        self._safe_ensure_gps_listeners()
        if self._curr_gps is None:
            self._gps_ready.wait(timeout=timeout)
        if self._gps_view is None:
            self._gps_view = LiveMappingView(self._gps_hub, self._gps_snapshot)
        return self._gps_view

    def get(self, *, timeout: Optional[float] = None) -> LiveMappingView:
        """Alias for :meth:`get_fix`."""
        return self.get_fix() if timeout is None else self.get_fix(timeout=timeout)

    def on_update(
        self, callback: "Callable[[dict[str, Any]], None]"
    ) -> StateSubscription:
        self._safe_ensure_gps_listeners()
        return self._gps_hub.subscribe(callback)
