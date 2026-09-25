"""WebRTC audio command handling for :class:`~cyberwave.driver.BaseDriver` subclasses.

Subscribe to ``start_audio`` / ``stop_audio`` on the twin MQTT command topic and
publish status responses. Pair with :class:`~cyberwave.driver.BaseDriver` and
the ``cyberwave`` client's twin audio streaming APIs.

Usage::

    from cyberwave.driver import BaseDriver, AudioStreamMixin

    class MyDriver(AudioStreamMixin, BaseDriver):
        async def on_register_callbacks(self) -> None:
            await self.register_audio_commands_async()

        async def _on_start_audio_async(self) -> None:
            await self._start_audio_streamer()

        async def _on_stop_audio_async(self) -> None:
            await self._audio_streamer.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

_AUDIO_TYPE_MAP: dict[str, str] = {
    "start_audio": "audio_started",
    "stop_audio": "audio_stopped",
}


class AudioStreamMixin:
    """Mixin for :class:`~cyberwave.driver.BaseDriver` subclasses that stream audio.

    Provides:

    - MQTT command subscription (``start_audio`` / ``stop_audio``)
    - Status response publishing (``{command}/status`` topics)
    - Lifecycle state transitions (``ACTIVE`` <-> ``INACTIVE``)
    - ``_audio_streaming_active`` state flag

    Implement :meth:`_on_start_audio_async` and :meth:`_on_stop_audio_async`
    using the bound twin / ``cyberwave`` client.

    Relies on :class:`~cyberwave.driver.BaseDriver`: ``_cw``, ``twin_uuid``,
    ``_mqtt_prefix``, :meth:`~cyberwave.driver.base.BaseDriver._transition_to`.
    """

    _audio_streaming_active: bool = False

    @abstractmethod
    async def _on_start_audio_async(self) -> None:
        """Start the audio stream."""
        ...

    @abstractmethod
    async def _on_stop_audio_async(self) -> None:
        """Stop the audio stream."""
        ...

    async def register_audio_commands_async(self) -> None:
        """Subscribe to ``start_audio`` / ``stop_audio`` on the twin command topic."""
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
                logger.warning("Failed to parse audio command payload: %s", exc)
                return
            if payload.get("source_type", "tele") != "tele":
                return
            command = payload.get("command")
            if command == "start_audio":
                asyncio.run_coroutine_threadsafe(self._handle_start_audio_async(), loop)
            elif command == "stop_audio":
                asyncio.run_coroutine_threadsafe(self._handle_stop_audio_async(), loop)

        cw.mqtt.subscribe(command_topic, _on_command)
        logger.info("Subscribed to audio command topic: %s", command_topic)

    def _publish_audio_status(self, command: str, status: str) -> None:
        """Publish an audio command status response to ``{command}/status``."""
        cw: Any = self._cw  # type: ignore[attr-defined]
        assert cw is not None
        prefix: str = self._mqtt_prefix  # type: ignore[attr-defined]
        twin_uuid: str = self.twin_uuid  # type: ignore[attr-defined]
        cw.mqtt.publish(
            f"{prefix}cyberwave/twin/{twin_uuid}/{command}/status",
            {
                "type": _AUDIO_TYPE_MAP[command],
                "status": status,
                "source_type": "edge",
                "timestamp": time.time(),
            },
        )

    async def _handle_start_audio_async(self) -> None:
        """Handle ``start_audio``: delegate to :meth:`_on_start_audio_async`, publish status."""
        from ..base import (
            DriverLifecycleState,
        )

        try:
            if not self._audio_streaming_active:
                logger.info("start_audio command received - starting stream...")
                await self._on_start_audio_async()
                self._audio_streaming_active = True
            else:
                logger.info("start_audio command received - stream already active")
            self._transition_to(DriverLifecycleState.ACTIVE)  # type: ignore[attr-defined]
            self._publish_audio_status("start_audio", "ok")
        except Exception:
            logger.exception("Failed to start audio stream on command")
            self._publish_audio_status("start_audio", "error")

    async def _handle_stop_audio_async(self) -> None:
        """Handle ``stop_audio``: delegate to :meth:`_on_stop_audio_async`, publish status."""
        from ..base import (
            DriverLifecycleState,
        )

        try:
            if self._audio_streaming_active:
                logger.info("stop_audio command received - stopping stream...")
                await self._on_stop_audio_async()
                self._audio_streaming_active = False
            else:
                logger.info("stop_audio command received - stream already stopped")
            self._transition_to(DriverLifecycleState.INACTIVE)  # type: ignore[attr-defined]
            self._publish_audio_status("stop_audio", "ok")
        except Exception:
            logger.exception("Failed to stop audio stream on command")
            self._publish_audio_status("stop_audio", "error")
