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

_VIDEO_COMMANDS_KEY = "video_commands"

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
    #: ``source_type`` values whose video commands are honoured (a missing
    #: ``source_type`` counts as ``tele``); ``None`` accepts every source.
    VIDEO_COMMAND_SOURCE_TYPES: frozenset[str] | None = frozenset({"tele"})
    #: Move the driver's lifecycle to ``ACTIVE`` / ``INACTIVE`` on start/stop.
    #: Right when the video is the device; a robot driver that also streams
    #: video sets ``False`` so stopping the video does not mark the robot inactive.
    VIDEO_COMMANDS_DRIVE_LIFECYCLE: bool = True
    _video_commands_registered: bool = False
    #: The last ``start_video`` / ``stop_video`` command payload, readable from
    #: :meth:`_on_start_video_async` (e.g. its ``data.recording`` flag).
    _video_command_payload: dict[str, Any] = {}

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

    def _video_command_needed(self, command: str) -> bool:
        """Whether *command* has work to do; ``False`` only acknowledges it.

        Default: from ``_streaming_active``. Override when the stream handlers
        are idempotent themselves (e.g. several streams behind one command).
        """
        return not self._streaming_active if command == "start_video" else self._streaming_active

    # ── Public interface ─────────────────────────────────────────────────────

    async def register_video_commands_async(self) -> None:
        """Subscribe to ``start_video`` / ``stop_video`` on the twin command topic.

        Call from :meth:`~cyberwave.driver.BaseDriver.on_register_callbacks`.
        Idempotent.
        """
        if self._video_commands_registered:
            return
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
            # By default only process commands from the frontend (source_type: "tele").
            allowed = type(self).VIDEO_COMMAND_SOURCE_TYPES
            if allowed is not None and payload.get("source_type", "tele") not in allowed:
                return
            # The payload rides with its own handler: a second command arriving
            # before the first runs must not retarget it.
            command = payload.get("command")
            if command == "start_video":
                asyncio.run_coroutine_threadsafe(self._handle_start_video_async(payload), loop)
            elif command == "stop_video":
                asyncio.run_coroutine_threadsafe(self._handle_stop_video_async(payload), loop)

        # Its own slot: the driver's command dispatch shares this topic, and
        # the default slot would replace it.
        cw.mqtt.subscribe(command_topic, _on_command, subscriber_key=_VIDEO_COMMANDS_KEY)
        self._video_commands_registered = True
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

    async def _handle_start_video_async(self, payload: dict[str, Any] | None = None) -> None:
        """Handle ``start_video``: delegate to :meth:`_on_start_video_async`, publish status."""
        if payload is not None:
            # Set on the loop thread, read by the hooks before their first await.
            self._video_command_payload = payload
        from ..base import (
            DriverLifecycleState,
        )  # avoid circular import at module level

        try:
            if self._video_command_needed("start_video"):
                logger.info("start_video command received — starting stream...")
                await self._on_start_video_async()
                self._streaming_active = True
            else:
                logger.info("start_video command received — stream already active")
            if type(self).VIDEO_COMMANDS_DRIVE_LIFECYCLE:
                self._transition_to(DriverLifecycleState.ACTIVE)  # type: ignore[attr-defined]
            self._publish_video_status("start_video", "ok")
        except Exception:
            logger.exception("Failed to start video stream on command")
            self._publish_video_status("start_video", "error")

    async def _handle_stop_video_async(self, payload: dict[str, Any] | None = None) -> None:
        """Handle ``stop_video``: delegate to :meth:`_on_stop_video_async`, publish status."""
        if payload is not None:
            # Set on the loop thread, read by the hooks before their first await.
            self._video_command_payload = payload
        from ..base import (
            DriverLifecycleState,
        )  # avoid circular import at module level

        try:
            if self._video_command_needed("stop_video"):
                logger.info("stop_video command received — stopping stream...")
                await self._on_stop_video_async()
                self._streaming_active = False
            else:
                logger.info("stop_video command received — stream already stopped")
            if type(self).VIDEO_COMMANDS_DRIVE_LIFECYCLE:
                self._transition_to(DriverLifecycleState.INACTIVE)  # type: ignore[attr-defined]
            self._publish_video_status("stop_video", "ok")
        except Exception:
            logger.exception("Failed to stop video stream on command")
            self._publish_video_status("stop_video", "error")
