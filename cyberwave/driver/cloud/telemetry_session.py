"""MQTT telemetry-session markers for a driver's connect/disconnect lifecycle.

:class:`TelemetrySessionMixin` publishes the ``telemetry_start`` / ``connected``
markers on connect and ``telemetry_end`` / ``disconnected`` on teardown, and
exposes :meth:`emit_driver_info` for queueing debounced ``driver_info`` fields.
Mixed into :class:`~cyberwave.driver.base.BaseDriver`.

**Host contract** — expects on ``self``: ``_cw`` (SDK client or None), ``_twin``,
``twin_uuid``, ``_telemetry`` (``BaseTelemetry``), ``_emit_driver_info(**fields)``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class TelemetrySessionMixin:
    """Session-marker publishing around the cloud connect/disconnect lifecycle."""

    def emit_driver_info(self, **fields: Any) -> None:
        """Queue fields for the next debounced ``driver_info`` telemetry publish."""
        self._emit_driver_info(**fields)

    def _start_driver_telemetry_session(self) -> None:
        """MQTT session markers via SDK client (not twin.telemetry high-level API)."""
        assert self._cw is not None
        twin_uuid = self.twin_uuid
        mqtt = self._cw.mqtt
        try:
            mqtt.publish_telemetry_start_message(twin_uuid)
        except Exception:
            logger.debug("publish_telemetry_start_message failed", exc_info=True)
        try:
            mqtt.publish_connected(twin_uuid)
        except Exception:
            logger.debug("publish_connected failed", exc_info=True)
        self._telemetry.mark_session_started()
        self._emit_driver_info()

    def _end_driver_telemetry_session(self) -> None:
        if self._cw is None or self._twin is None:
            return
        twin_uuid = self._twin.uuid
        mqtt = self._cw.mqtt
        self._telemetry.publish_if_dirty()
        self._telemetry.mark_session_ended()
        try:
            mqtt.publish_telemetry_end(twin_uuid)
        except Exception:
            logger.debug("publish_telemetry_end failed", exc_info=True)
        try:
            mqtt.publish_disconnected(twin_uuid)
        except Exception:
            logger.debug("publish_disconnected failed", exc_info=True)
