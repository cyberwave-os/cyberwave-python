"""WebRTC video command handling for :class:`~cyberwave.driver.BaseDriver` subclasses.

Subscribe to ``start_video`` / ``stop_video`` on the twin MQTT command topic and
publish status responses. Pair with :class:`~cyberwave.driver.BaseDriver` and
the ``cyberwave`` client's twin streaming APIs.

Usage::

    from cyberwave.driver import BaseDriver, VideoStreamMixin

    class MyDriver(VideoStreamMixin, BaseDriver):
        async def on_register_callbacks(self) -> None:
            await self.register_video_commands_async()

        async def _on_start_video_async(self) -> None:
            await self._twin.stream_video_background(...)

        async def _on_stop_video_async(self) -> None:
            await self._twin.stop_streaming()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

_VIDEO_TYPE_MAP: dict[str, str] = {
    "start_video": "video_started",
    "stop_video": "video_stopped",
}


class VideoStreamMixin:
    """Mixin for :class:`~cyberwave.driver.BaseDriver` subclasses that stream video.

    Provides:

    - MQTT command subscription (``start_video`` / ``stop_video``)
    - Status response publishing (``{command}/status`` topics)
    - Lifecycle state transitions (``ACTIVE`` ↔ ``INACTIVE``)
    - ``_streaming_active`` state flag

    The concrete driver must implement :meth:`_on_start_video_async` and
    :meth:`_on_stop_video_async` using the bound twin / ``cyberwave`` client.

    Relies on attributes from :class:`~cyberwave.driver.BaseDriver`: ``_cw``,
    ``twin_uuid``, ``_mqtt_prefix``, :meth:`~cyberwave.driver.base.BaseDriver._transition_to`.
    """

    # Class-level default; becomes an instance attribute on first assignment.
    _streaming_active: bool = False

    # ── Abstract hooks ───────────────────────────────────────────────────────

    @abstractmethod
    async def _on_start_video_async(self) -> None:
        """Start the video stream.

        Called by :meth:`_handle_start_video_async` when a ``start_video``
        command is received and the stream is not yet active.  Raise an
        exception to signal failure — the mixin will publish an error status.
        """
        ...

    @abstractmethod
    async def _on_stop_video_async(self) -> None:
        """Stop the video stream.

        Called by :meth:`_handle_stop_video_async` when a ``stop_video``
        command is received and the stream is currently active.  Raise an
        exception to signal failure — the mixin will publish an error status.
        """
        ...

    # ── Public interface ─────────────────────────────────────────────────────

    async def register_video_commands_async(self) -> None:
        """Subscribe to ``start_video`` / ``stop_video`` on the twin command topic.

        Call from :meth:`~cyberwave.driver.BaseDriver.on_register_callbacks`.
        """
        cw: Any = self._cw  # type: ignore[attr-defined]
        assert cw is not None
        loop = asyncio.get_running_loop()
        prefix: str = self._mqtt_prefix  # type: ignore[attr-defined]
        twin_uuid: str = self.twin_uuid  # type: ignore[attr-defined]
        command_topic = f"{prefix}cyberwave/twin/{twin_uuid}/command"

        def _on_command(data: Any) -> None:
            try:
                payload = data if isinstance(data, dict) else json.loads(data)
            except Exception as exc:
                logger.warning("Failed to parse command payload: %s", exc)
                return
            # Only process commands from the frontend (source_type: "tele").
            if payload.get("source_type", "tele") != "tele":
                return
            command = payload.get("command")
            if command == "start_video":
                asyncio.run_coroutine_threadsafe(self._handle_start_video_async(), loop)
            elif command == "stop_video":
                asyncio.run_coroutine_threadsafe(self._handle_stop_video_async(), loop)

        cw.mqtt.subscribe(command_topic, _on_command)
        logger.info("Subscribed to video command topic: %s", command_topic)

    def _publish_video_status(self, command: str, status: str) -> None:
        """Publish a video command status response to ``{command}/status``.

        Args:
            command: ``"start_video"`` or ``"stop_video"``
            status:  ``"ok"`` or ``"error"``
        """
        cw: Any = self._cw  # type: ignore[attr-defined]
        assert cw is not None
        prefix: str = self._mqtt_prefix  # type: ignore[attr-defined]
        twin_uuid: str = self.twin_uuid  # type: ignore[attr-defined]
        cw.mqtt.publish(
            f"{prefix}cyberwave/twin/{twin_uuid}/{command}/status",
            {
                "type": _VIDEO_TYPE_MAP[command],
                "status": status,
                "source_type": "edge",
                "timestamp": time.time(),
            },
        )

    # ── Internal handlers ─────────────────────────────────────────────────────

    async def _handle_start_video_async(self) -> None:
        """Handle ``start_video``: delegate to :meth:`_on_start_video_async`, publish status."""
        from ..base import (
            DriverLifecycleState,
        )  # avoid circular import at module level

        try:
            if not self._streaming_active:
                logger.info("start_video command received — starting stream...")
                await self._on_start_video_async()
                self._streaming_active = True
            else:
                logger.info("start_video command received — stream already active")
            self._transition_to(DriverLifecycleState.ACTIVE)  # type: ignore[attr-defined]
            self._publish_video_status("start_video", "ok")
        except Exception:
            logger.exception("Failed to start video stream on command")
            self._publish_video_status("start_video", "error")

    async def _handle_stop_video_async(self) -> None:
        """Handle ``stop_video``: delegate to :meth:`_on_stop_video_async`, publish status."""
        from ..base import (
            DriverLifecycleState,
        )  # avoid circular import at module level

        try:
            if self._streaming_active:
                logger.info("stop_video command received — stopping stream...")
                await self._on_stop_video_async()
                self._streaming_active = False
            else:
                logger.info("stop_video command received — stream already stopped")
            self._transition_to(DriverLifecycleState.INACTIVE)  # type: ignore[attr-defined]
            self._publish_video_status("stop_video", "ok")
        except Exception:
            logger.exception("Failed to stop video stream on command")
            self._publish_video_status("stop_video", "error")
