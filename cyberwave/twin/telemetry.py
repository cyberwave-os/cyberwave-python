"""Twin-scoped telemetry facade (transport publish only in v1)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import Twin

class TwinTelemetry:
    """High-level telemetry handle for a :class:`~cyberwave.twin.base.Twin`.

    Only :meth:`publish` is available from the twin side; it sends an ad-hoc
    payload on ``cyberwave/twin/{uuid}/telemetry``.  Lifecycle helpers
    (session start/end, driver info) are intended for edge drivers — use
    :class:`~cyberwave.driver.base.BaseDriver` or the MQTT client directly
    from driver code.
    """

    def __init__(self, twin: Twin) -> None:
        self._twin = twin

    def publish(self, payload: dict[str, Any]) -> None:
        """Publish a payload on ``cyberwave/twin/{uuid}/telemetry``."""
        self._twin.publish_telemetry(payload)

    @property
    def is_connected(self) -> bool:
        raise NotImplementedError(
            "twin.telemetry.is_connected is not yet implemented."
        )

    def set_connected(self, connected: bool, **extra: Any) -> None:
        raise NotImplementedError(
            "twin.telemetry.set_connected() is not yet implemented."
        )

    def session_start(self, **extra: Any) -> None:
        raise NotImplementedError(
            "twin.telemetry.session_start() is not yet implemented."
        )

    def session_end(self, **extra: Any) -> None:
        raise NotImplementedError(
            "twin.telemetry.session_end() is not yet implemented."
        )

    def driver_info(self, info: dict[str, Any] | None = None, **kwargs: Any) -> None:
        raise NotImplementedError(
            "twin.telemetry.driver_info() is not yet implemented."
        )

    def update_snapshot(self, **fields: Any) -> None:
        raise NotImplementedError(
            "twin.telemetry.update_snapshot() is not yet implemented."
        )

    def flush_snapshot(self, event_type: str = "driver_info") -> None:
        raise NotImplementedError(
            "twin.telemetry.flush_snapshot() is not yet implemented."
        )

    @staticmethod
    def standard_driver_info(**fields: Any) -> dict[str, Any]:
        """Build canonical ``driver_info`` fields for :class:`~cyberwave.driver.base.BaseDriver`."""
        return {k: v for k, v in fields.items() if v is not None}
