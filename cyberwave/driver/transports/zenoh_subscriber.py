"""Imperative Zenoh command subscribe for Python SDK drivers.

Receive autonomous commands (e.g. from an edge worker) on Zenoh ``commands/*``
channels while keeping teleop priority. For registry-declared command channels,
prefer :class:`~cyberwave.driver.interface.registry_mixin.InterfaceRegistryMixin`
on :class:`~cyberwave.driver.BaseDriver`.

Usage::

    from cyberwave.driver import BaseDriver, ZenohPublisherMixin, ZenohSubscriberMixin

    class MyDriver(BaseDriver, ZenohPublisherMixin, ZenohSubscriberMixin):
        async def on_register_callbacks(self) -> None:
            self._init_zenoh_command_bus(
                twin_uuid=self.twin_uuid,
                cw_client=self._cw,
                loop=asyncio.get_running_loop(),
            )
            self._register_command_handler(
                "commands/velocity",
                self._on_zenoh_velocity_command,
                watchdog_ms=500,
            )

        async def on_shutdown(self) -> None:
            await self._close_zenoh_command_bus()

        async def _on_zenoh_velocity_command(self, payload: dict, ctx: CommandContext) -> None:
            vx = float(payload.get("linear_x", 0.0))
            angular = float(payload.get("angular_z", 0.0))
            await self._sport_controller.send_velocity(vx, 0.0, angular)

        def _is_teleop_active(self) -> bool:
            # Called by the mixin before dispatching a command.
            return time.monotonic() < self._teleop_active_until

Safety guarantees
-----------------
* **Command watchdog**: if no message arrives within *watchdog_ms* on a
  registered channel, ``on_command_timeout(channel)`` is called on the driver
  (default: log a warning).  Override to implement a safety stop.
* **Teleop priority**: before dispatching any command, the mixin calls
  ``_is_teleop_active()`` on the driver (default: ``False``).  Override to
  implement a lock-out window when a human operator is in control.
* **Thread safety**: Zenoh callbacks arrive on a background thread; the mixin
  bridges them into the driver's asyncio event loop via
  ``asyncio.run_coroutine_threadsafe``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from cyberwave.data.api import DataBus

logger = logging.getLogger(__name__)

# Type alias: async callable that receives (payload, context)
CommandHandler = Callable[["dict[str, Any]", "CommandContext"], Awaitable[None]]


@dataclass(frozen=True)
class CommandContext:
    """Metadata passed alongside every command payload to the handler."""

    channel: str
    """The channel name, e.g. ``"commands/velocity"``."""
    twin_uuid: str
    """The twin UUID this command targets."""
    received_at: float = field(default_factory=time.monotonic)
    """``time.monotonic()`` stamp at reception (for latency measurement)."""


class _ChannelSubscription:
    """Internal bookkeeping for a single channel subscription."""

    def __init__(
        self,
        channel: str,
        handler: CommandHandler,
        watchdog_ms: int,
        loop: asyncio.AbstractEventLoop,
        watchdog_callback: Callable[[str], None],
    ) -> None:
        self.channel = channel
        self.handler = handler
        self.watchdog_ms = watchdog_ms
        self._loop = loop
        self._watchdog_callback = watchdog_callback
        self._last_received: float = time.monotonic()
        self._armed: bool = False  # True after the first message arrives
        self._subscription: Any = None  # DataBus Subscription handle
        self._watchdog_thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start_watchdog(self) -> None:
        if self.watchdog_ms <= 0:
            return
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
            name=f"zenoh-cmd-watchdog-{self.channel}",
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        interval = self.watchdog_ms / 1000.0
        while not self._stop.is_set():
            self._stop.wait(interval)
            if self._stop.is_set():
                break
            if not self._armed:
                continue
            elapsed_ms = (time.monotonic() - self._last_received) * 1000
            if elapsed_ms >= self.watchdog_ms:
                self._watchdog_callback(self.channel)

    def touch(self) -> None:
        """Record that a message was just received."""
        self._last_received = time.monotonic()
        self._armed = True

    def stop(self) -> None:
        self._stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=1.0)
            self._watchdog_thread = None


class ZenohSubscriberMixin:
    """Mixin for Zenoh command subscription via the SDK :class:`~cyberwave.data.api.DataBus`.

    Use with :class:`~cyberwave.driver.BaseDriver`: register channels in
    :meth:`~cyberwave.driver.BaseDriver.on_register_callbacks` and close in
    :meth:`~cyberwave.driver.BaseDriver.on_shutdown`.

    Attributes set by ``_init_zenoh_command_bus()``:
        _zenoh_cmd_bus:     DataBus | None
        _zenoh_cmd_subs:    dict[str, _ChannelSubscription]
        _zenoh_cmd_loop:    asyncio.AbstractEventLoop | None
        _zenoh_cmd_twin:    str
    """

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def _init_zenoh_command_bus(
        self,
        *,
        twin_uuid: str,
        loop: asyncio.AbstractEventLoop,
        cw_client: Any = None,
    ) -> None:
        """Initialise the Zenoh command subscriber bus.

        Call from :meth:`~cyberwave.driver.BaseDriver.on_register_callbacks`.

        Args:
            twin_uuid:  UUID of the digital twin.
            loop:       Running asyncio event loop (``asyncio.get_running_loop()``).
            cw_client:  Optional :class:`~cyberwave.Cyberwave` client. When provided,
                        uses ``cw_client.data_bus_for(twin_uuid)``. When ``None``,
                        ``DataBus`` is built from env vars.
        """
        self._zenoh_cmd_bus: DataBus | None = None
        self._zenoh_cmd_bus_owned: bool = False  # True when we built the DataBus ourselves
        self._zenoh_cmd_subs: dict[str, _ChannelSubscription] = {}
        self._zenoh_cmd_loop: asyncio.AbstractEventLoop | None = loop
        self._zenoh_cmd_twin: str = twin_uuid

        if not os.getenv("CYBERWAVE_DATA_BACKEND"):
            logger.debug(
                "[ZENOH-CMD] CYBERWAVE_DATA_BACKEND not set; command bus disabled."
            )
            return

        try:
            if cw_client is not None:
                data_bus: DataBus = cw_client.data_bus_for(twin_uuid)
                owns_bus = False
            else:
                from cyberwave.data.api import DataBus as _DataBus  # type: ignore[import-untyped]
                from cyberwave.data.config import get_backend  # type: ignore[import-untyped]

                data_bus = _DataBus(get_backend(), twin_uuid=twin_uuid)
                owns_bus = True

            self._zenoh_cmd_bus = data_bus
            self._zenoh_cmd_bus_owned = owns_bus
            logger.info(
                "[ZENOH-CMD] Command bus active (twin=%s, backend=%s)",
                twin_uuid,
                type(data_bus._backend).__name__,
            )
        except Exception as exc:
            logger.warning(
                "[ZENOH-CMD] Failed to init command bus (continuing without): %s", exc
            )

    def _register_command_handler(
        self,
        channel: str,
        handler: CommandHandler,
        *,
        watchdog_ms: int = 500,
    ) -> None:
        """Subscribe to a command channel and register an async handler.

        Args:
            channel:      Channel name, e.g. ``"commands/velocity"``.
            handler:      Async callable ``(payload: dict, ctx: CommandContext)``.
                          Called on the driver's asyncio event loop.
            watchdog_ms:  Milliseconds of silence before ``on_command_timeout``
                          is called.  Set to ``0`` to disable the watchdog.
        """
        if self._zenoh_cmd_bus is None:
            logger.debug(
                "[ZENOH-CMD] Command bus not active; skipping registration of %s", channel
            )
            return
        if self._zenoh_cmd_loop is None:
            logger.warning("[ZENOH-CMD] No event loop; cannot register %s", channel)
            return

        loop = self._zenoh_cmd_loop
        twin_uuid = self._zenoh_cmd_twin

        sub_state = _ChannelSubscription(
            channel=channel,
            handler=handler,
            watchdog_ms=watchdog_ms,
            loop=loop,
            watchdog_callback=self._on_command_timeout_internal,
        )

        def _on_decoded(decoded: Any) -> None:
            """Called from the Zenoh background thread with already-decoded data."""
            sub_state.touch()

            if not isinstance(decoded, dict):
                logger.warning(
                    "[ZENOH-CMD] Expected dict on %s, got %s",
                    channel,
                    type(decoded).__name__,
                )
                return

            if self._is_teleop_active():
                logger.debug(
                    "[ZENOH-CMD] Teleop active — dropping command on %s", channel
                )
                return

            ctx = CommandContext(channel=channel, twin_uuid=twin_uuid)
            asyncio.run_coroutine_threadsafe(handler(decoded, ctx), loop)

        subscription = self._zenoh_cmd_bus.subscribe(channel, _on_decoded)
        sub_state._subscription = subscription
        sub_state.start_watchdog()
        self._zenoh_cmd_subs[channel] = sub_state

        logger.info(
            "[ZENOH-CMD] Registered handler for %s (watchdog=%dms)", channel, watchdog_ms
        )

    async def _close_zenoh_command_bus(self) -> None:
        """Unsubscribe all channels, stop watchdogs, and close the bus.

        Call from ``on_shutdown()`` (BaseDriver).  Idempotent.
        """
        for sub in self._zenoh_cmd_subs.values():
            sub.stop()
            if sub._subscription is not None:
                with contextlib.suppress(Exception):
                    sub._subscription.close()
        self._zenoh_cmd_subs.clear()

        if self._zenoh_cmd_bus is not None and self._zenoh_cmd_bus_owned:
            with contextlib.suppress(Exception):
                self._zenoh_cmd_bus.close()
        self._zenoh_cmd_bus = None

    # ── Hooks (override in driver) ─────────────────────────────────────────

    def _is_teleop_active(self) -> bool:
        """Return ``True`` if a human teleop operator currently has priority.

        Override in the driver to implement a lock-out window::

            def _is_teleop_active(self) -> bool:
                return time.monotonic() < self._teleop_active_until

        Default: always returns ``False`` (autonomous commands are dispatched).
        """
        return False

    def on_command_timeout(self, channel: str) -> None:
        """Called when no command has been received on *channel* for ``watchdog_ms``.

        The default implementation logs a warning.  Override to implement a
        safety stop::

            def on_command_timeout(self, channel: str) -> None:
                asyncio.run_coroutine_threadsafe(
                    self._sport_controller.send_velocity(0.0, 0.0, 0.0),
                    self._zenoh_cmd_loop,
                )
        """
        logger.warning(
            "[ZENOH-CMD] Watchdog timeout on channel '%s' — no command received.", channel
        )

    # ── Internal ───────────────────────────────────────────────────────────

    def _on_command_timeout_internal(self, channel: str) -> None:
        """Internal watchdog callback — delegates to ``on_command_timeout``."""
        try:
            self.on_command_timeout(channel)
        except Exception as exc:
            logger.warning(
                "[ZENOH-CMD] Exception in on_command_timeout for %s: %s", channel, exc
            )
