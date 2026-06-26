"""Thread-safe command inbox for cross-thread driver command hand-off.

Drivers receive commands on one thread (e.g. the MQTT/asyncio loop) and apply
them on another (e.g. the ROS executor in ``tick()``). ``CommandInbox`` owns the
lock, the merge, and the staleness guard so each driver does not hand-roll them.

Staleness policy (fixes the bug where missing timestamps bypassed dedup and
never advanced the watermark, and equal timestamps were dropped):

- A timestamped message **older** than the last accepted timestamp is dropped.
- A timestamped message **equal to or newer** is accepted and advances the
  watermark.
- An **untimestamped** message is always accepted (relaxed) and leaves the
  watermark unchanged.
"""

from __future__ import annotations

import threading
from typing import Any


class CommandInbox:
    """A merge inbox: producers ``submit`` updates, a consumer ``drain``s them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, Any] = {}
        self._last_timestamp: float | None = None

    def submit(self, update: dict[str, Any], *, timestamp: float | None = None) -> bool:
        """Merge *update* into the pending set. Returns False if dropped as stale."""
        with self._lock:
            if (
                timestamp is not None
                and self._last_timestamp is not None
                and timestamp < self._last_timestamp
            ):
                return False
            if timestamp is not None:
                self._last_timestamp = timestamp
            self._pending.update(update)
            return True

    def drain(self) -> dict[str, Any]:
        """Return a copy of the pending updates and clear the inbox."""
        with self._lock:
            pending = dict(self._pending)
            self._pending.clear()
            return pending

    def clear(self) -> None:
        """Discard pending updates without touching the staleness watermark."""
        with self._lock:
            self._pending.clear()

    def reset(self) -> None:
        """Clear pending updates and the watermark (e.g. on entering NO_OP)."""
        with self._lock:
            self._pending.clear()
            self._last_timestamp = None

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)
