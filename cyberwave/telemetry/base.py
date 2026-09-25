"""Driver/edge telemetry snapshot collector (health-monitor style).

Modeled after :class:`~cyberwave.edge.health.EdgeHealthCheck`: periodic snapshot
+ provider callbacks, with append-only event records and a non-blocking public
API. Publishing is delegated to an injected transport hook — typically
:meth:`~cyberwave.twin.base.Twin.publish_telemetry`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

TelemetrySnapshotProvider = Callable[[], Mapping[str, Any]]
TelemetryPublishHook = Callable[[dict[str, Any]], None]


@dataclass
class TelemetryEvent:
    """Single telemetry state change (for debugging / future MQTT export)."""

    timestamp: float
    event_type: str
    fields: dict[str, Any] = field(default_factory=dict)


class BaseTelemetry:
    """Collect driver telemetry snapshots; publish via an injected hook.

    Drivers wire a :class:`BaseTelemetry` at init and register a registry
    publisher callback (``twin/telemetry``) that calls :meth:`publish_if_dirty`.
    High-level session APIs (``telemetry_start`` / ``connected`` on the twin
    handle) remain on the MQTT client; this class owns debounced ``driver_info``
    payloads only.
    """

    def __init__(
        self,
        *,
        publish_payload: TelemetryPublishHook,
        snapshot_provider: TelemetrySnapshotProvider | None = None,
        source_type: str = "edge",
        max_events: int = 64,
    ) -> None:
        self._publish_payload = publish_payload
        self._snapshot_provider = snapshot_provider
        self._source_type = source_type
        self._pending: dict[str, Any] = {}
        self._dirty = False
        self._events: list[TelemetryEvent] = []
        self._max_events = max_events
        self._session_active = False

    @property
    def session_active(self) -> bool:
        return self._session_active

    def mark_session_started(self) -> None:
        self._session_active = True
        self.update(session="active")

    def mark_session_ended(self) -> None:
        self._session_active = False
        self.update(session="ended")

    def update(self, **fields: Any) -> None:
        """Queue fields for the next ``driver_info`` publish (marks dirty)."""
        for key, value in fields.items():
            if value is not None:
                self._pending[key] = value
        if fields:
            self._dirty = True

    def record_event(self, event_type: str, **fields: Any) -> None:
        """Append a telemetry event and merge *fields* into the pending snapshot."""
        self._events.append(
            TelemetryEvent(timestamp=time.time(), event_type=event_type, fields=dict(fields))
        )
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events :]
        self.update(**fields)

    def build_payload(self, event_type: str = "driver_info") -> dict[str, Any]:
        """Build a publish-ready MQTT payload (does not send)."""
        payload: dict[str, Any] = {
            "type": event_type,
            "timestamp": time.time(),
            "source_type": self._source_type,
        }
        if self._snapshot_provider is not None:
            try:
                snap = self._snapshot_provider()
                if isinstance(snap, Mapping):
                    payload.update(dict(snap))
            except Exception as exc:
                logger.debug("telemetry snapshot_provider failed: %s", exc)
        payload.update(self._pending)
        return payload

    def publish_if_dirty(self, *, event_type: str = "driver_info") -> bool:
        """Publish when :meth:`update` has been called since the last flush."""
        if not self._dirty:
            return False
        self._publish_payload(self.build_payload(event_type))
        self._pending.clear()
        self._dirty = False
        return True

    def publish_now(self, event_type: str, **fields: Any) -> None:
        """Publish immediately (bypasses dirty debounce)."""
        payload = self.build_payload(event_type)
        payload.update(fields)
        self._publish_payload(payload)

    def recent_events(self) -> list[TelemetryEvent]:
        """Return a copy of recent :class:`TelemetryEvent` records."""
        return list(self._events)
