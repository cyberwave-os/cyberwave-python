"""Twin-scoped telemetry facade (transport publish only in v1)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import Twin

_NOT_IMPL = (
    "Not implemented on twin.telemetry in v1. "
    "Drivers use cyberwave.telemetry.BaseTelemetry + a registry publisher on "
    "twin/telemetry; session start/end use client.mqtt.publish_telemetry_* . "
    "For ad-hoc payloads call twin.telemetry.publish({...}) or twin.publish_telemetry(...)."
)


class TwinTelemetry:
    """High-level telemetry handle for a :class:`~cyberwave.twin.base.Twin`.

  In v1 only :meth:`publish` is implemented; it delegates to
  :meth:`~cyberwave.twin.base.Twin.publish_telemetry` (transport layer).
  Lifecycle/session helpers raise :exc:`NotImplementedError` until a future
  SDK release wires them through the same transport path.
    """

    def __init__(self, twin: Twin) -> None:
        self._twin = twin

    def publish(self, payload: dict[str, Any]) -> None:
        """Publish a payload on ``cyberwave/twin/{uuid}/telemetry``."""
        self._twin.publish_telemetry(payload)

    @property
    def is_connected(self) -> bool:
        raise NotImplementedError(_NOT_IMPL)

    def set_connected(self, connected: bool, **extra: Any) -> None:
        raise NotImplementedError(_NOT_IMPL)

    def session_start(self, **extra: Any) -> None:
        raise NotImplementedError(_NOT_IMPL)

    def session_end(self, **extra: Any) -> None:
        raise NotImplementedError(_NOT_IMPL)

    def driver_info(self, info: dict[str, Any] | None = None, **kwargs: Any) -> None:
        raise NotImplementedError(_NOT_IMPL)

    def update_snapshot(self, **fields: Any) -> None:
        raise NotImplementedError(_NOT_IMPL)

    def flush_snapshot(self, event_type: str = "driver_info") -> None:
        raise NotImplementedError(_NOT_IMPL)

    @staticmethod
    def standard_driver_info(**fields: Any) -> dict[str, Any]:
        """Build canonical ``driver_info`` fields for :class:`~cyberwave.driver.base.BaseDriver`."""
        return {k: v for k, v in fields.items() if v is not None}
