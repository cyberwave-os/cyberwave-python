"""Callback hub + subscription — the shared MQTT-consumption callback core.

One callback registry type (`CallbackHub`) and one subscription type
(`StateSubscription`) for the whole SDK. Both the numpy snapshot path
(`consumers.mqtt_snapshot.MqttSensorStreamHandle`) and the dict/pose live-view
path (`consumers.mqtt_live_view`) fan out through this.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

__all__ = ["StateSubscription", "CallbackHub"]


class StateSubscription:
    """Handle returned by ``subscribe``/``on_update``; ``cancel()`` stops delivery."""

    def __init__(self, hub: "CallbackHub", key: int) -> None:
        self._hub = hub
        self._key = key
        self._active = True

    def cancel(self) -> None:
        if self._active:
            self._hub._remove(self._key)
            self._active = False


class CallbackHub:
    """Thread-safe callback registry for one live-state handle/stream."""

    def __init__(self, *, label: str = "state") -> None:
        self._label = label
        self._callbacks: dict[int, Callable[[Any], None]] = {}
        self._seq = 0
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[Any], None]) -> StateSubscription:
        with self._lock:
            key = self._seq
            self._seq += 1
            self._callbacks[key] = callback
        return StateSubscription(self, key)

    def _remove(self, key: int) -> None:
        with self._lock:
            self._callbacks.pop(key, None)

    def notify(self, snapshot_factory: Callable[[], Any]) -> None:
        """Invoke every subscriber with a fresh snapshot (one per callback)."""
        with self._lock:
            callbacks = list(self._callbacks.values())
        for cb in callbacks:
            try:
                cb(snapshot_factory())
            except Exception:
                logger.exception("%s on_update callback raised; ignoring", self._label)
