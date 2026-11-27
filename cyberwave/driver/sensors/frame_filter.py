"""Fail-closed substitution of worker-processed frames into a camera stream.

A worker (for example an anonymiser) publishes processed frames on
:data:`cyberwave.data.FILTERED_FRAME_CHANNEL` (``frames/filtered``). A camera
driver with the filter enabled encodes those instead of the raw capture.

Contract:

* The processed frame must match the raw frame's shape and dtype.
* When no fresh, matching processed frame is available, the driver encodes a
  black frame. There is no raw fallback: to see what the camera sees, disable
  the filter.
* Blank frames are summarised once per :data:`STALE_LOG_S` window, so operators
  can tell freshness jitter (~1%) from a dead worker (~100%).

:class:`~cyberwave.driver.sensors.camera.CameraStreamMixin` wires the
subscription; this module has no SDK dependency so it unit-tests in isolation.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

# Tuning knobs — defaults. ``FRESHNESS_MS`` is tunable per-driver via the
# ``freshness_ms`` constructor kwarg (the driver wires this up from its
# own env-var parsing; see the driver's module docs). ``STALE_LOG_S`` is
# a log-hygiene decision (the rolling-window length), not an operational
# knob, and is intentionally not exposed.
# 200 ms ≈ 6 frames at 30 fps; large enough to absorb a worker hiccup,
# small enough that prolonged staleness flips to blank within ~one cycle.
FRESHNESS_MS = 200.0
STALE_LOG_S = 30.0
# Above this fraction of blank frames in a window the operator almost
# certainly has a dead worker rather than freshness-window jitter, so
# the log message switches to a more pointed advice line.
HIGH_BLANK_RATE = 0.9


class ProcessedFrameSlot:
    """Thread-safe single-slot store for the latest processed frame.

    Updated by the Zenoh subscriber thread; read by the driver's frame
    callback (which runs on whatever thread the SDK dispatches from).
    """

    def __init__(self) -> None:
        self._frame: np.ndarray | None = None
        self._timestamp: float = 0.0
        self._lock = threading.Lock()

    def put(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self._timestamp = time.monotonic()

    def get_if_fresh(self, max_age_s: float) -> np.ndarray | None:
        """Return the latest frame if it arrived within *max_age_s* seconds.

        Returns ``None`` if the slot is empty or the cached frame is stale.
        """
        with self._lock:
            if self._frame is None:
                return None
            if time.monotonic() - self._timestamp > max_age_s:
                return None
            return self._frame

    @property
    def timestamp(self) -> float:
        """Monotonic timestamp of the latest ``put()``. ``0.0`` if empty."""
        with self._lock:
            return self._timestamp

    @timestamp.setter
    def timestamp(self, value: float) -> None:
        """Override the cached timestamp (used by tests to simulate staleness)."""
        with self._lock:
            self._timestamp = value


class _BlankFrameStats:
    """Rolling window of :meth:`FrameFilter.apply` outcomes.

    The window starts at construction and rolls forward whenever
    :meth:`expired` returns True. Counters are intentionally plain ints
    (not threading-protected); ``apply`` is only invoked from the
    driver's frame callback thread, so there is exactly one writer.
    """

    def __init__(self, *, window_s: float, now: float) -> None:
        self.window_s = window_s
        self.start_ts = now
        self.total = 0
        self.blank = 0
        self.reasons: dict[str, int] = {}

    def record_pass(self) -> None:
        self.total += 1

    def record_blank(self, reason: str) -> None:
        self.total += 1
        self.blank += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def expired(self, now: float) -> bool:
        if self.window_s <= 0:
            return False
        return (now - self.start_ts) >= self.window_s

    def reset(self, now: float) -> None:
        self.start_ts = now
        self.total = 0
        self.blank = 0
        self.reasons = {}


class FrameFilter:
    """Per-driver state container for the frame-filter feature.

    Owns the latest-frame slot and the rolling blank-frame stats. The
    driver is responsible for actually subscribing to Zenoh and feeding
    processed frames into :meth:`store_processed`.

    The ``freshness_ms`` and ``stale_log_s`` constructor parameters
    default to the module constants. Drivers override ``freshness_ms``
    from their own env-var parsing (e.g.
    ``CYBERWAVE_METADATA_FRAME_FILTER_FRESHNESS_MS``) — this module
    intentionally stays env-free so it's trivially unit-testable without
    the SDK. ``stale_log_s`` is a log-hygiene knob (window length); set
    to ``0`` (or any negative value) to disable summary logging.
    """

    def __init__(
        self,
        *,
        channel: str | None,
        freshness_ms: float = FRESHNESS_MS,
        stale_log_s: float = STALE_LOG_S,
    ) -> None:
        self.channel = channel
        self.freshness_s = max(0.0, freshness_ms / 1000.0)
        self.stale_log_s = max(0.0, stale_log_s)
        self._slot = ProcessedFrameSlot()
        self._stats = _BlankFrameStats(window_s=self.stale_log_s, now=time.monotonic())

    @property
    def enabled(self) -> bool:
        return self.channel is not None

    @property
    def slot(self) -> ProcessedFrameSlot:
        return self._slot

    def store_processed(self, frame: object) -> None:
        """Zenoh subscriber callback target. Silently drops non-ndarray samples.

        Shape / dtype validation happens in :meth:`apply` — at this point
        we don't yet have the raw frame to compare against.
        """
        if not isinstance(frame, np.ndarray):
            logger.debug(
                "[FRAME_FILTER] Ignoring non-ndarray sample (type=%s)",
                type(frame).__name__,
            )
            return
        self._slot.put(frame)

    def apply(self, raw_frame: np.ndarray) -> np.ndarray | None:
        """Decide what frame the SDK should encode for *raw_frame*.

        Returns:
            * ``None`` to keep the raw frame (filter disabled).
            * A new ``ndarray`` to replace the raw frame (the worker's
              processed frame when fresh and shape/dtype-matched,
              otherwise a black frame).
        """
        if not self.enabled:
            return None
        try:
            processed = self._slot.get_if_fresh(self.freshness_s)
            if processed is None:
                self._stats.record_blank("stale")
                return np.zeros_like(raw_frame)
            if processed.shape != raw_frame.shape or processed.dtype != raw_frame.dtype:
                # A worker publishing the wrong shape/dtype would corrupt the
                # WebRTC encoder — fail closed and account it as a mismatch.
                self._stats.record_blank("shape_mismatch")
                return np.zeros_like(raw_frame)
            self._stats.record_pass()
            return processed
        finally:
            # Window flush runs after every apply so we always log a summary
            # promptly once the window expires, regardless of the next apply
            # rate (the driver may stop streaming for unrelated reasons).
            self._maybe_flush_window(time.monotonic())

    def _maybe_flush_window(self, now: float) -> None:
        """If the rolling window has elapsed, log a summary and reset."""
        if not self._stats.expired(now):
            return
        if self._stats.blank > 0:
            elapsed = max(now - self._stats.start_ts, self._stats.window_s)
            pct = 100.0 * self._stats.blank / max(self._stats.total, 1)
            reason_summary = ", ".join(
                f"{reason}={count}" for reason, count in sorted(self._stats.reasons.items())
            )
            if pct >= HIGH_BLANK_RATE * 100.0:
                advice = (
                    "Worker likely down or not publishing on this channel — "
                    "check the worker container logs."
                )
            else:
                advice = (
                    "Worker may be slow or freshness window too tight; consider "
                    "raising CYBERWAVE_METADATA_FRAME_FILTER_FRESHNESS_MS if "
                    "pixelation occasionally drops out."
                )
            logger.warning(
                "[FRAME_FILTER] %.1f%% of %d frames in last %.0f s emitted blank "
                "fallback on '%s' (reasons: %s). %s",
                pct,
                self._stats.total,
                elapsed,
                self.channel,
                reason_summary or "n/a",
                advice,
            )
        self._stats.reset(now)
