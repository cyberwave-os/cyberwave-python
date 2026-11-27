"""Event-loop lag probe for the SDK's WebRTC streamers.

aiortc answers STUN consent checks, sends RTP and runs the reconnect monitor
on the asyncio loop. If per-frame work (capture, encoding, overlays) blocks
that loop, consent answers arrive late and the connection can close on its
own. The probe makes that visible: once a second it compares when it was
scheduled to wake with when it actually woke, and logs a WARNING when the lag
crosses a threshold, at most once per reporting window, with the largest lag
seen in that window.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

#: How often the probe wakes, in seconds.
LOOP_LAG_INTERVAL_S = 1.0
#: Lag above which a wake-up counts as late, in seconds.
LOOP_LAG_WARN_S = 0.25
#: Minimum time between two warnings, in seconds.
LOOP_LAG_REPORT_EVERY_S = 30.0


class EventLoopLagProbe:
    """Measure how late the event loop wakes a 1 s timer; warn when it lags.

    Args:
        label: Names the stream in the warning (e.g. ``twin <uuid> camera <name>``).
        interval: Seconds between wake-ups.
        threshold: Lag in seconds above which a wake-up is late.
        report_every: Minimum seconds between warnings.
        clock: Monotonic clock, injectable for tests.
    """

    def __init__(
        self,
        label: str,
        *,
        interval: float = LOOP_LAG_INTERVAL_S,
        threshold: float = LOOP_LAG_WARN_S,
        report_every: float = LOOP_LAG_REPORT_EVERY_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.label = label
        self.interval = interval
        self.threshold = threshold
        self.report_every = report_every
        self._clock = clock
        self._handle: Optional[asyncio.TimerHandle] = None
        self._due = 0.0
        self._last_report: Optional[float] = None
        self._window_max = 0.0
        self._window_late = 0
        #: Largest lag seen since the probe started, in seconds.
        self.max_lag = 0.0

    @property
    def running(self) -> bool:
        return self._handle is not None

    def start(self) -> None:
        """Start the probe on the running loop (no-op if already running).

        The probe is a chain of ``loop.call_later`` timers, not a task awaiting
        ``asyncio.sleep``: it never holds the loop, and code (or a test) that
        replaces ``asyncio.sleep`` cannot turn it into a busy loop.
        """
        if self.running:
            return
        self._schedule(asyncio.get_running_loop())

    async def stop(self) -> None:
        """Stop the probe; no timer fires after this returns."""
        handle, self._handle = self._handle, None
        if handle is not None:
            handle.cancel()

    def record(self, lag: float) -> None:
        """Account one wake-up that was ``lag`` seconds late."""
        self.max_lag = max(self.max_lag, lag)
        if lag <= self.threshold:
            return
        self._window_max = max(self._window_max, lag)
        self._window_late += 1
        now = self._clock()
        if self._last_report is not None and now - self._last_report < self.report_every:
            return
        logger.warning(
            "Event loop lag for %s: woke %.0f ms late (max %.0f ms over %d late "
            "wake-ups since the last report). A blocked loop delays WebRTC "
            "consent answers and RTP.",
            self.label,
            lag * 1000,
            self._window_max * 1000,
            self._window_late,
        )
        self._last_report = now
        self._window_max = 0.0
        self._window_late = 0

    def _schedule(self, loop: asyncio.AbstractEventLoop) -> None:
        self._due = self._clock() + self.interval
        self._handle = loop.call_later(self.interval, self._tick, loop)

    def _tick(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._handle is None:  # stopped
            return
        self.record(max(0.0, self._clock() - self._due))
        if loop.is_closed():
            self._handle = None
            return
        self._schedule(loop)
