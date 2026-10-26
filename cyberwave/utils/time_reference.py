"""Thread-safe wall + monotonic clock used to synchronise multi-stream timestamps."""

from __future__ import annotations

import sys
import threading
import time


def _select_monotonic_clock() -> tuple[str, bool]:
    """Pick the monotonic clock to stamp acquisition from, preferring suspend-inclusive.

    ``time.monotonic()`` stops while the host is suspended on both platforms, which
    breaks the (wall, monotonic) offset by the whole sleep. ``CLOCK_BOOTTIME`` is the
    variant that continues on Linux; on Darwin that is ``CLOCK_MONOTONIC``. Probed
    rather than assumed, because the constant is importable if the *build* host had
    the clock. See "Acquisition stamps come from a suspend-inclusive clock" in
    ``cyberwave-sdks/README.md``.
    """
    candidate = "CLOCK_MONOTONIC" if sys.platform == "darwin" else "CLOCK_BOOTTIME"
    clock_id = getattr(time, candidate, None)
    if clock_id is not None:
        try:
            time.clock_gettime(clock_id)
        except OSError:
            pass
        else:
            return candidate, True
    return "monotonic", False


MONOTONIC_CLOCK_NAME, MONOTONIC_CLOCK_IS_SUSPEND_INCLUSIVE = _select_monotonic_clock()

_MONOTONIC_CLOCK_ID: int | None = (
    getattr(time, MONOTONIC_CLOCK_NAME)
    if MONOTONIC_CLOCK_IS_SUSPEND_INCLUSIVE
    else None
)


def monotonic_seconds() -> float:
    """Read the acquisition clock named by :data:`MONOTONIC_CLOCK_NAME`.

    Use this rather than ``time.monotonic()`` for anything stamping acquisition;
    deadlines and retry backoff should keep the plain clock.
    """
    if _MONOTONIC_CLOCK_ID is None:
        return time.monotonic()
    return time.clock_gettime(_MONOTONIC_CLOCK_ID)


class TimeReference:
    """Thread-safe time reference for coordinating timestamps across multiple streams.

    The main control loop should call :meth:`update` periodically (e.g. at 100 Hz) to
    refresh the cached wall and monotonic clock. Other threads (e.g. camera tracks)
    call :meth:`read` for lock-free access to the last values written by the loop.

    Camera streamers that are not given a ``TimeReference`` should use
    ``time.time()`` / :func:`monotonic_seconds` at capture time instead (handled
    inside the sensor stack, not by callers of this class).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._time = time.time()
        self._time_monotonic = monotonic_seconds()

    def update(self) -> tuple[float, float]:
        """Refresh cached time from the OS. Intended for the main control loop only.

        Returns:
            Tuple of (wall_clock_timestamp, monotonic_timestamp) just written.
        """
        with self._lock:
            self._time = time.time()
            self._time_monotonic = monotonic_seconds()
            return self._time, self._time_monotonic

    def read(self) -> tuple[float, float]:
        """Return the last cached (wall, monotonic) pair from :meth:`update`.

        Does not acquire the lock; safe paired with :meth:`update` on one writer thread.
        """
        return self._time, self._time_monotonic
