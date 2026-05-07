import threading
import time


class TimeReference:
    """Thread-safe time reference for coordinating timestamps across multiple streams.

    The main control loop should call :meth:`update` periodically (e.g. at 100 Hz) to
    refresh the cached wall and monotonic clock. Other threads (e.g. camera tracks)
    call :meth:`read` for lock-free access to the last values written by the loop.

    Camera streamers that are not given a ``TimeReference`` should use
    ``time.time()`` / ``time.monotonic()`` at capture time instead (handled inside
    the sensor stack, not by callers of this class).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._time = time.time()
        self._time_monotonic = time.monotonic()

    def update(self) -> tuple[float, float]:
        """Refresh cached time from the OS. Intended for the main control loop only.

        Returns:
            Tuple of (wall_clock_timestamp, monotonic_timestamp) just written.
        """
        with self._lock:
            self._time = time.time()
            self._time_monotonic = time.monotonic()
            return self._time, self._time_monotonic

    def read(self) -> tuple[float, float]:
        """Return the last cached (wall, monotonic) pair from :meth:`update`.

        Does not acquire the lock; safe paired with :meth:`update` on one writer thread.
        """
        return self._time, self._time_monotonic
