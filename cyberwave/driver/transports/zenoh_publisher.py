"""Imperative Zenoh publishing for Python SDK drivers (frames and JSON channels).

Use when you need high-rate binary frames or ad-hoc channel publish outside the
interface registry. For tick-based JSON streams (IMU, joint_states), prefer
``TopicSpec(enable_zenoh=True)`` on
:class:`~cyberwave.driver.BaseDriver` — the registry opens the DataBus automatically.

Usage (:class:`~cyberwave.driver.BaseDriver`)::

    from cyberwave.driver import BaseDriver, ZenohPublisherMixin

    class MyDriver(BaseDriver, ZenohPublisherMixin):
        async def on_activate(self) -> None:
            self._init_zenoh_bus(
                twin_uuid=self.twin_uuid,
                cw_client=self._cw,
            )

        async def on_shutdown(self) -> None:
            self._close_zenoh_bus()

Edge-runtime ROS2 drivers can mix this in as well (sync ``activate`` / ``shutdown`` hooks).
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from cyberwave.data.api import DataBus

logger = logging.getLogger(__name__)


class _FrameSlot:
    """Single-slot swap buffer: producer (device callback) -> consumer (Zenoh thread).

    ``put()`` is O(1) pointer swap; safe from any thread or asyncio coroutine.
    ``take()`` blocks up to *timeout* seconds; returns ``None`` on timeout.

    When the producer is faster than the consumer, older frames are silently
    replaced -- semantically equivalent to the ``latest`` Zenoh subscriber
    policy.
    """

    def __init__(self) -> None:
        self._frame: np.ndarray | None = None
        self._event = threading.Event()
        self._lock = threading.Lock()

    def put(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
        self._event.set()

    def take(self, timeout: float = 1.0) -> np.ndarray | None:
        self._event.wait(timeout)
        self._event.clear()
        with self._lock:
            frame = self._frame
            self._frame = None
        return frame


class ZenohPublisherMixin:
    """Mixin for imperative Zenoh publish via the SDK :class:`~cyberwave.data.api.DataBus`.

    Compatible with :class:`~cyberwave.driver.BaseDriver` (async hooks). Pass
    *cw_client* from the bound twin so the bus uses ``data_bus_for(twin_uuid)``
    (do not rely on ``CYBERWAVE_TWIN_UUID`` env alone).

    Attributes set by ``_init_zenoh_bus()``:
        _zenoh_data_bus:    DataBus | None
        _zenoh_frame_slot:  _FrameSlot | None
        _zenoh_stop:        threading.Event
        _zenoh_thread:      threading.Thread | None
        _zenoh_channel:     str  (frame channel name)
        _zenoh_frame_metadata: dict | None  (metadata bound on the first frame)
    """

    def _init_zenoh_bus(
        self,
        *,
        twin_uuid: str,
        cw_client: Any = None,
        frame_channel: str = "frames/default",
        frame_metadata: dict[str, Any] | None = {"color_format": "rgb24"},
    ) -> None:
        """Initialize the Zenoh data bus and frame publisher thread.

        Call from :meth:`~cyberwave.driver.BaseDriver.on_activate` (or sync
        ``activate()`` in edge-runtime drivers).

        Args:
            twin_uuid:      UUID of the digital twin.
            cw_client:      Optional :class:`~cyberwave.Cyberwave` client. When
                            provided, prefer ``cw_client.data_bus_for(twin_uuid)``.
                            When ``None``, ``DataBus`` is built from env vars.
            frame_channel:  Zenoh channel for binary frames.
            frame_metadata: Metadata bound to *frame_channel* on the first frame
                            (see ``HeaderTemplate`` -- metadata is immutable after
                            the first publish). Defaults to ``rgb24`` for camera
                            callers; pass ``None`` (or a suitable dict) for
                            non-RGB frames such as uint16 depth.
        """
        self._zenoh_data_bus: DataBus | None = None
        self._zenoh_frame_slot: _FrameSlot | None = None
        self._zenoh_stop: threading.Event = threading.Event()
        self._zenoh_thread: threading.Thread | None = None
        self._zenoh_channel: str = frame_channel
        self._zenoh_frame_metadata: dict[str, Any] | None = frame_metadata

        if not os.getenv("CYBERWAVE_DATA_BACKEND"):
            return

        try:
            from cyberwave.data.config import is_zenoh_publish_enabled  # type: ignore[import-untyped]  # noqa: I001

            if not is_zenoh_publish_enabled():
                mode = os.environ.get("CYBERWAVE_PUBLISH_MODE", "dual")
                logger.debug(
                    "[ZENOH] Zenoh publishing disabled via CYBERWAVE_PUBLISH_MODE=%s; skipping data bus init.",
                    mode,
                )
                return

            if cw_client is not None:
                data_bus: DataBus = cw_client.data_bus_for(twin_uuid)
            else:
                from cyberwave.data.api import DataBus as _DataBus  # type: ignore[import-untyped] # noqa: I001
                from cyberwave.data.config import get_backend  # type: ignore[import-untyped]

                data_bus = _DataBus(get_backend(), twin_uuid=twin_uuid)

            self._zenoh_data_bus = data_bus
            self._zenoh_frame_slot = _FrameSlot()
            self._zenoh_thread = threading.Thread(
                target=self._zenoh_frame_loop,
                daemon=True,
                name=f"zenoh-frames-{twin_uuid[:8]}",
            )
            self._zenoh_thread.start()

            backend_name = type(data_bus._backend).__name__
            logger.info(
                "[ZENOH] Publisher active (twin=%s, channel=%s, backend=%s)",
                twin_uuid,
                frame_channel,
                backend_name,
            )
        except Exception as exc:
            logger.warning("[ZENOH] Failed to init data bus (continuing without): %s", exc)
            self._zenoh_data_bus = None
            self._zenoh_frame_slot = None

    def _zenoh_frame_loop(self) -> None:
        """Publisher thread: drain ``_FrameSlot`` and call ``DataBus.publish``."""
        _first = True
        while not self._zenoh_stop.is_set():
            if self._zenoh_frame_slot is None or self._zenoh_data_bus is None:
                time.sleep(0.1)
                continue
            frame = self._zenoh_frame_slot.take(timeout=1.0)
            if frame is None:
                continue
            try:
                if _first:
                    self._zenoh_data_bus.publish(
                        self._zenoh_channel,
                        frame,
                        metadata=self._zenoh_frame_metadata,
                    )
                    _first = False
                else:
                    self._zenoh_data_bus.publish(self._zenoh_channel, frame)
            except Exception as exc:
                logger.warning(
                    "[ZENOH] Frame publish failed on %s: %s", self._zenoh_channel, exc
                )

    def zenoh_publish_frame(self, frame: np.ndarray) -> None:
        """Non-blocking frame publish -- swaps a pointer in < 1 us.

        Safe to call from asyncio coroutines, ROS2 callbacks, or any thread.
        No-op if Zenoh is not configured.
        """
        if self._zenoh_frame_slot is not None:
            self._zenoh_frame_slot.put(frame)

    def zenoh_publish(
        self,
        channel: str,
        payload: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Publish any payload type to a Zenoh channel with full error isolation.

        Accepts ``numpy.ndarray``, ``dict``, or ``bytes`` -- delegates to
        ``DataBus.publish()``.  MQTT and other paths are unaffected if this
        raises.

        *metadata* is forwarded to ``DataBus.publish()`` and cached on first
        call per channel (see ``HeaderTemplate``).
        """
        if self._zenoh_data_bus is None:
            return
        try:
            self._zenoh_data_bus.publish(channel, payload, metadata=metadata)
        except Exception as exc:
            logger.warning("[ZENOH] Publish failed on %s: %s", channel, exc)

    def _close_zenoh_bus(self) -> None:
        """Stop the frame publisher thread and close the ``DataBus``.

        Call from :meth:`~cyberwave.driver.BaseDriver.on_shutdown`. Idempotent.
        """
        if hasattr(self, "_zenoh_stop"):
            self._zenoh_stop.set()
        if hasattr(self, "_zenoh_thread") and self._zenoh_thread is not None:
            self._zenoh_thread.join(timeout=3.0)
            self._zenoh_thread = None
        if hasattr(self, "_zenoh_data_bus") and self._zenoh_data_bus is not None:
            with contextlib.suppress(Exception):
                self._zenoh_data_bus.close()
            self._zenoh_data_bus = None
