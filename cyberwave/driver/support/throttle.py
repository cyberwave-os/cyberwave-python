"""Throttle — fire at most once per interval, always on the first call.

Replaces hand-rolled ``time.monotonic()`` elapsed-checks. The clock is injectable
so callers can test deterministically without patching the process clock.
"""

from __future__ import annotations

import time
from collections.abc import Callable


class Throttle:
    """Rate gate: ``ready()`` is True on the first call and once ``interval_s`` has
    elapsed since the last True. The last-fire time advances only when True is returned."""

    def __init__(
        self, interval_s: float, *, now: Callable[[], float] = time.monotonic
    ) -> None:
        self._interval_s = interval_s
        self._now = now
        self._last_fire: float | None = None

    def ready(self) -> bool:
        now = self._now()
        if self._last_fire is None or now - self._last_fire >= self._interval_s:
            self._last_fire = now
            return True
        return False
