"""Per-channel ring buffer with O(log n) time-indexed lookups.

``TimeIndexedRingBuffer`` stores a fixed number of ``(timestamp, value)``
entries in a :class:`collections.deque` (O(1) eviction) and supports:

* **Insertion** — ``append(ts, value)`` adds a sample, automatically evicting
  the oldest when capacity is exceeded.
* **Point query** — ``at(t)`` returns the two bounding entries for
  interpolation (or an exact match).
* **Range query** — ``window(from_t, to_t)`` returns all entries whose
  timestamps fall within ``[from_t, to_t]``.
* **Binary search** — all lookups are O(log n) via :func:`bisect`.

Thread safety: reads and writes are serialised via a :class:`threading.Lock`.
The lock is fine-grained (per-buffer, not global) and held only for the
duration of the in-memory operation — no I/O under the lock.
"""

from __future__ import annotations

import threading
from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")

DEFAULT_CAPACITY = 1000


@dataclass(slots=True, frozen=True)
class TimestampedSample(Generic[T]):
    """A value tagged with its acquisition timestamp."""

    ts: float
    value: T


@dataclass(slots=True, frozen=True)
class BracketResult(Generic[T]):
    """Result of a point query that lands between two samples.

    ``before`` and ``after`` are the bounding entries.  Either may be
    ``None`` when the query timestamp falls outside the buffered range.
    ``exact`` is set when the query timestamp matches a sample exactly.
    """

    before: TimestampedSample[T] | None = None
    after: TimestampedSample[T] | None = None
    exact: TimestampedSample[T] | None = None


class TimeIndexedRingBuffer(Generic[T]):
    """Fixed-capacity ring buffer with O(log n) timestamp-indexed queries.

    Samples **must** be appended in non-decreasing timestamp order.  Out-of-
    order inserts raise :class:`ValueError`.

    Parameters
    ----------
    capacity:
        Maximum number of samples retained.  When exceeded, the oldest
        sample is evicted.  Must be >= 1.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._timestamps: deque[float] = deque(maxlen=capacity)
        self._values: deque[T] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    # -- Properties -----------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Maximum number of samples the buffer can hold."""
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return len(self._timestamps)

    @property
    def empty(self) -> bool:
        with self._lock:
            return len(self._timestamps) == 0

    @property
    def oldest_ts(self) -> float | None:
        """Timestamp of the oldest buffered sample, or ``None``."""
        with self._lock:
            return self._timestamps[0] if self._timestamps else None

    @property
    def newest_ts(self) -> float | None:
        """Timestamp of the newest buffered sample, or ``None``."""
        with self._lock:
            return self._timestamps[-1] if self._timestamps else None

    # -- Mutation -------------------------------------------------------------

    def append(self, ts: float, value: T) -> None:
        """Insert a sample.  The oldest entry is evicted automatically when at
        capacity (O(1) via :class:`collections.deque`).

        Equal timestamps are allowed — multiple samples may share the same
        timestamp.  In that case, :meth:`at` returns the *first* sample
        inserted at that timestamp.

        Raises
        ------
        ValueError
            If *ts* is less than the newest buffered timestamp (out-of-order).
        """
        with self._lock:
            if self._timestamps and ts < self._timestamps[-1]:
                raise ValueError(
                    f"Out-of-order insert: ts={ts} < newest={self._timestamps[-1]}"
                )
            self._timestamps.append(ts)
            self._values.append(value)

    def clear(self) -> None:
        """Remove all buffered samples."""
        with self._lock:
            self._timestamps.clear()
            self._values.clear()

    # -- Queries --------------------------------------------------------------

    def at(self, t: float) -> BracketResult[T]:
        """Find the bounding samples for timestamp *t*.

        Returns a :class:`BracketResult` with:

        * ``exact`` set when a sample exists at exactly *t*.
        * ``before`` / ``after`` set to the surrounding samples for
          interpolation.
        * Both ``before`` and ``after`` ``None`` when the buffer is empty.
        * ``before`` ``None`` when *t* is before all buffered samples.
        * ``after`` ``None`` when *t* is after all buffered samples.
        """
        with self._lock:
            n = len(self._timestamps)
            if n == 0:
                return BracketResult()

            idx = bisect_left(self._timestamps, t)

            if idx < n and self._timestamps[idx] == t:
                return BracketResult(
                    exact=TimestampedSample(self._timestamps[idx], self._values[idx]),
                )

            before: TimestampedSample[T] | None = None
            after: TimestampedSample[T] | None = None

            if idx > 0:
                before = TimestampedSample(
                    self._timestamps[idx - 1], self._values[idx - 1]
                )
            if idx < n:
                after = TimestampedSample(self._timestamps[idx], self._values[idx])

            return BracketResult(before=before, after=after)

    def window(
        self,
        from_t: float,
        to_t: float,
    ) -> list[TimestampedSample[T]]:
        """Return all samples with timestamps in ``[from_t, to_t]``.

        Returns an empty list when the range contains no samples.

        Raises
        ------
        ValueError
            If *from_t* > *to_t*.
        """
        if from_t > to_t:
            raise ValueError(f"from_t ({from_t}) must be <= to_t ({to_t})")

        with self._lock:
            if not self._timestamps:
                return []

            lo = bisect_left(self._timestamps, from_t)
            hi = bisect_right(self._timestamps, to_t)

            return [
                TimestampedSample(self._timestamps[i], self._values[i])
                for i in range(lo, hi)
            ]

    def latest(self) -> TimestampedSample[T] | None:
        """Return the newest sample, or ``None`` if empty."""
        with self._lock:
            if not self._timestamps:
                return None
            return TimestampedSample(self._timestamps[-1], self._values[-1])

    def all_samples(self) -> list[TimestampedSample[T]]:
        """Return a snapshot of all buffered samples (oldest first)."""
        with self._lock:
            return [
                TimestampedSample(ts, val)
                for ts, val in zip(self._timestamps, self._values)
            ]

    # -- Repr -----------------------------------------------------------------

    def __repr__(self) -> str:
        with self._lock:
            n = len(self._timestamps)
        return f"TimeIndexedRingBuffer(capacity={self._capacity}, size={n})"
