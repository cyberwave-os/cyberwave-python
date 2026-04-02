"""Time-aware sensor fusion primitives — ``data.at()`` and ``data.window()``.

These APIs sit on top of per-channel :class:`TimeIndexedRingBuffer` instances
and provide:

* **Point reads** — ``at(channel, t=, interpolation=)`` returns a value
  interpolated to an arbitrary timestamp.  Linear interpolation is used for
  scalar / vector channels, SLERP for quaternion channels.
* **Range queries** — ``window(channel, from_t=, to_t=)`` or
  ``window(channel, duration_ms=)`` returns all buffered samples in a time
  range.

Both primitives share the same per-channel ring buffer, so enabling
``data.at()`` does not double memory usage compared to ``data.window()``.

Typical usage (inside a worker callback)::

    joints = cw.data.at("joint_states", t=ctx.timestamp, interpolation="linear")
    imu = cw.data.window("imu", from_t=prev_ts, to_t=ctx.timestamp)
    recent_ft = cw.data.window("force_torque", duration_ms=100)
"""

from __future__ import annotations

import math
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from .ring_buffer import (
    BracketResult,
    TimeIndexedRingBuffer,
    TimestampedSample,
)

try:
    import numpy as _np
except ImportError:
    _np = None

VALID_INTERPOLATIONS = ("linear", "slerp", "nearest", "none")

__all__ = [
    "WindowResult",
    "Quaternion",
    "ChannelBuffer",
    "FusionLayer",
    "interpolate_linear",
    "interpolate_slerp",
    "interpolate_nearest",
    "VALID_INTERPOLATIONS",
    "INTERPOLATION_STRATEGIES",
]


@dataclass(slots=True, frozen=True)
class WindowResult:
    """Result of a ``window()`` query.

    ``samples`` is a chronologically ordered list of
    :class:`TimestampedSample` entries whose timestamps fall within the
    requested range.
    """

    samples: list[TimestampedSample[Any]]
    from_t: float
    to_t: float

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[TimestampedSample[Any]]:
        return iter(self.samples)

    def __bool__(self) -> bool:
        return len(self.samples) > 0

    @property
    def values(self) -> list[Any]:
        """Extract just the values from all samples."""
        return [s.value for s in self.samples]

    @property
    def timestamps(self) -> list[float]:
        """Extract just the timestamps from all samples."""
        return [s.ts for s in self.samples]


@dataclass(slots=True, frozen=True)
class Quaternion:
    """Unit quaternion for type-safe SLERP dispatch.

    Wrapping orientation data in ``Quaternion`` ensures that
    ``interpolation="slerp"`` applies spherical interpolation.  Plain
    ``list[float]`` of length 4 will **not** trigger SLERP — it must
    be an explicit ``Quaternion`` instance.

    Convention: Hamilton ``(x, y, z, w)`` — the same as ROS, MuJoCo,
    and the Cyberwave wire format.  Defaults to the identity rotation.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def as_list(self) -> list[float]:
        """Return ``[x, y, z, w]``."""
        return [self.x, self.y, self.z, self.w]


# ---------------------------------------------------------------------------
# Low-level interpolation helpers
# ---------------------------------------------------------------------------


def _lerp_scalar(a: float, b: float, alpha: float) -> float:
    """Linear interpolation between two scalars."""
    return a + alpha * (b - a)


def _lerp_list(a: list[float], b: list[float], alpha: float) -> list[float]:
    """Element-wise linear interpolation between two lists.

    Warns when the lists differ in length; interpolation proceeds over
    the common prefix (shorter list length).
    """
    if len(a) != len(b):
        warnings.warn(
            f"List length mismatch during interpolation ({len(a)} vs {len(b)}); "
            "interpolating over common prefix only.",
            UserWarning,
            stacklevel=3,
        )
    return [ai + alpha * (bi - ai) for ai, bi in zip(a, b)]


def _warn_schema_drift(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    stacklevel: int = 5,
) -> None:
    """Emit a warning when *a* and *b* have different key sets.

    *stacklevel* is forwarded to :func:`warnings.warn` so callers can
    tune which frame appears in the warning message.
    """
    a_keys, b_keys = set(a), set(b)
    if a_keys != b_keys:
        parts: list[str] = []
        only_a = a_keys - b_keys
        only_b = b_keys - a_keys
        if only_a:
            parts.append(f"only in before: {sorted(only_a)}")
        if only_b:
            parts.append(f"only in after: {sorted(only_b)}")
        warnings.warn(
            f"Schema drift during dict interpolation ({'; '.join(parts)})",
            UserWarning,
            stacklevel=stacklevel,
        )


def _normalize_quaternion(q: list[float]) -> list[float]:
    """Normalise a quaternion [x, y, z, w] to unit length."""
    norm = math.sqrt(sum(c * c for c in q))
    if norm < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [c / norm for c in q]


def _dot_quaternion(a: list[float], b: list[float]) -> float:
    """Dot product of two quaternions."""
    return sum(ai * bi for ai, bi in zip(a, b))


def _slerp_quaternion(a: list[float], b: list[float], alpha: float) -> list[float]:
    """Spherical linear interpolation between two unit quaternions [x,y,z,w].

    Handles the antipodal case (negative dot product) by flipping ``b``.
    Falls back to linear interpolation for nearly-parallel quaternions to
    avoid division by a near-zero sine.
    """
    a = _normalize_quaternion(a)
    b = _normalize_quaternion(b)

    dot = _dot_quaternion(a, b)

    if dot < 0.0:
        b = [-c for c in b]
        dot = -dot

    dot = min(dot, 1.0)

    if dot > 0.9995:
        result = [ai + alpha * (bi - ai) for ai, bi in zip(a, b)]
        return _normalize_quaternion(result)

    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    wa = math.sin((1.0 - alpha) * theta) / sin_theta
    wb = math.sin(alpha * theta) / sin_theta

    return _normalize_quaternion([wa * ai + wb * bi for ai, bi in zip(a, b)])


def _quat_nlerp(va: Quaternion, vb: Quaternion, alpha: float) -> Quaternion:
    """Normalized linear interpolation (NLERP) between two quaternions.

    Cheaper than SLERP but produces non-constant angular velocity for large
    angle differences.  Used by ``interpolation="linear"`` on quaternion data.
    """
    lerped = _lerp_list(va.as_list(), vb.as_list(), alpha)
    return Quaternion(*_normalize_quaternion(lerped))


def _quat_slerp_fn(va: Quaternion, vb: Quaternion, alpha: float) -> Quaternion:
    """Spherical linear interpolation (SLERP) between two quaternions."""
    return Quaternion(*_slerp_quaternion(va.as_list(), vb.as_list(), alpha))


def _interp_dict(
    a: dict[str, Any],
    b: dict[str, Any],
    alpha: float,
    quat_fn: Callable[[Quaternion, Quaternion, float], Quaternion],
    *,
    stacklevel: int = 4,
) -> dict[str, Any]:
    """Interpolate two dicts key-by-key.

    Quaternion-valued keys are interpolated with *quat_fn*; numeric scalars
    and lists use :func:`_lerp_scalar` / :func:`_lerp_list`; all other values
    are passed through from *a* (the "before" sample).  Warns on schema drift.

    *stacklevel* controls which call frame appears in :func:`warnings.warn`
    output.  It is propagated to :func:`_warn_schema_drift` incremented by one
    to account for this function's own frame.
    """
    _warn_schema_drift(a, b, stacklevel=stacklevel + 1)
    result: dict[str, Any] = {}
    for key in set(a) | set(b):
        if key not in a:
            result[key] = b[key]
            continue
        if key not in b:
            result[key] = a[key]
            continue
        va, vb = a[key], b[key]
        if isinstance(va, Quaternion) and isinstance(vb, Quaternion):
            result[key] = quat_fn(va, vb, alpha)
        elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            result[key] = _lerp_scalar(float(va), float(vb), alpha)
        elif isinstance(va, list) and isinstance(vb, list):
            if va and isinstance(va[0], (int, float)):
                result[key] = _lerp_list(
                    [float(x) for x in va],
                    [float(x) for x in vb],
                    alpha,
                )
            else:
                result[key] = va
        else:
            result[key] = va
    return result


def _lerp_dict(a: dict[str, Any], b: dict[str, Any], alpha: float) -> dict[str, Any]:
    """Linear interpolation for dict values.  Quaternion fields use NLERP."""
    return _interp_dict(a, b, alpha, _quat_nlerp, stacklevel=5)


def _slerp_dict(a: dict[str, Any], b: dict[str, Any], alpha: float) -> dict[str, Any]:
    """SLERP interpolation for dict values.  Quaternion fields use SLERP."""
    return _interp_dict(a, b, alpha, _quat_slerp_fn, stacklevel=4)


# ---------------------------------------------------------------------------
# Public interpolation strategies
# ---------------------------------------------------------------------------


def interpolate_linear(
    before: TimestampedSample[Any],
    after: TimestampedSample[Any],
    t: float,
) -> Any:
    """Linear interpolation dispatcher.

    Handles: ``float``, ``int``, ``list[float]``, ``dict`` with numeric
    values, and ``numpy.ndarray`` (element-wise lerp).

    Note on quaternions: ``Quaternion``-valued channels use **NLERP**
    (normalized linear interpolation) rather than SLERP.  NLERP is cheaper
    but produces non-constant angular velocity for large angle differences.
    Use ``interpolation="slerp"`` when accurate orientation interpolation is
    required.
    """
    if before.ts == after.ts:
        return before.value

    alpha = (t - before.ts) / (after.ts - before.ts)
    va, vb = before.value, after.value

    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
        return _lerp_scalar(float(va), float(vb), alpha)

    if isinstance(va, Quaternion) and isinstance(vb, Quaternion):
        return _quat_nlerp(va, vb, alpha)

    if isinstance(va, list) and isinstance(vb, list):
        if va and isinstance(va[0], (int, float)):
            return _lerp_list(
                [float(x) for x in va],
                [float(x) for x in vb],
                alpha,
            )
        return va

    if isinstance(va, dict) and isinstance(vb, dict):
        return _lerp_dict(va, vb, alpha)

    if _np is not None and isinstance(va, _np.ndarray) and isinstance(vb, _np.ndarray):
        return va + alpha * (vb - va)

    return va


def interpolate_slerp(
    before: TimestampedSample[Any],
    after: TimestampedSample[Any],
    t: float,
) -> Any:
    """SLERP interpolation for :class:`Quaternion`-valued channels.

    Values (or dict fields) must be :class:`Quaternion` instances for
    SLERP to apply.  Plain ``list[float]`` of length 4 will **not**
    trigger SLERP — wrap them in ``Quaternion(x, y, z, w)`` first.

    Falls back to linear interpolation (with a warning) for
    non-quaternion types.
    """
    if before.ts == after.ts:
        return before.value

    alpha = (t - before.ts) / (after.ts - before.ts)
    va, vb = before.value, after.value

    if isinstance(va, Quaternion) and isinstance(vb, Quaternion):
        return _quat_slerp_fn(va, vb, alpha)

    if isinstance(va, dict) and isinstance(vb, dict):
        return _slerp_dict(va, vb, alpha)

    warnings.warn(
        f"slerp interpolation requested but values are not quaternion-like "
        f"(got {type(before.value).__name__}); falling back to linear",
        UserWarning,
        stacklevel=3,
    )
    return interpolate_linear(before, after, t)


def interpolate_nearest(
    before: TimestampedSample[Any],
    after: TimestampedSample[Any],
    t: float,
) -> Any:
    """Return the sample closest in time to *t*."""
    if before.ts == after.ts:
        return before.value
    d_before = abs(t - before.ts)
    d_after = abs(t - after.ts)
    return before.value if d_before <= d_after else after.value


INTERPOLATION_STRATEGIES = {
    "linear": interpolate_linear,
    "slerp": interpolate_slerp,
    "nearest": interpolate_nearest,
}


# ---------------------------------------------------------------------------
# ChannelBuffer — per-channel ring buffer fed by subscriptions
# ---------------------------------------------------------------------------


class ChannelBuffer:
    """Manages a :class:`TimeIndexedRingBuffer` for a single channel.

    Typically created and owned by :class:`FusionLayer`.

    Thread safety
    -------------
    Concurrent reads and writes are safe: all mutation and query operations
    delegate to :class:`~cyberwave.data.ring_buffer.TimeIndexedRingBuffer`,
    which serialises access with its own internal lock.
    """

    def __init__(self, channel: str, capacity: int) -> None:
        self.channel = channel
        self.buffer: TimeIndexedRingBuffer[Any] = TimeIndexedRingBuffer(capacity)

    def ingest(self, ts: float, value: Any) -> None:
        """Append a decoded sample into the ring buffer."""
        self.buffer.append(ts, value)

    def at(
        self,
        t: float,
        interpolation: str = "linear",
    ) -> Any | None:
        """Interpolated point read at timestamp *t*.

        Returns ``None`` when the buffer is empty.  When *t* falls outside
        the buffered range the nearest boundary sample is returned
        (constant extrapolation).

        Raises
        ------
        ValueError
            If *interpolation* is not one of the recognised strategies.
        """
        if interpolation not in VALID_INTERPOLATIONS:
            raise ValueError(
                f"Invalid interpolation '{interpolation}'. "
                f"Must be one of: {', '.join(repr(i) for i in VALID_INTERPOLATIONS)}"
            )

        bracket: BracketResult[Any] = self.buffer.at(t)

        if bracket.exact is not None:
            return bracket.exact.value

        if bracket.before is None and bracket.after is None:
            return None

        if interpolation == "none":
            return None

        if bracket.before is None:
            return bracket.after.value if bracket.after else None

        if bracket.after is None:
            return bracket.before.value

        strategy = INTERPOLATION_STRATEGIES.get(interpolation)
        if strategy is None:
            return bracket.before.value

        return strategy(bracket.before, bracket.after, t)

    def window(
        self,
        *,
        from_t: float,
        to_t: float,
    ) -> WindowResult:
        """Return all buffered samples in ``[from_t, to_t]``.

        Use :meth:`FusionLayer.window` if you need ``duration_ms`` support
        with a custom clock.

        Raises
        ------
        ValueError
            If *from_t* > *to_t* (delegated from the ring buffer).
        """
        samples = self.buffer.window(from_t, to_t)
        return WindowResult(samples=samples, from_t=from_t, to_t=to_t)


# ---------------------------------------------------------------------------
# FusionLayer — owns all channel buffers, exposes at() and window()
# ---------------------------------------------------------------------------


class FusionLayer:
    """Manages per-channel ring buffers for time-aware fusion.

    Created and owned by the ``DataBus`` (CYB-1554).  Workers interact
    with it indirectly through ``cw.data.at()`` and ``cw.data.window()``.

    Parameters
    ----------
    default_capacity:
        Default ring-buffer depth for channels without an explicit
        :meth:`configure_channel` call.
    clock:
        Callable returning the current time as a ``float``.  Defaults to
        :func:`time.time`.  Override with e.g. a monotonic clock or
        ROS ``sim_time`` to match whatever clock source your samples use.

    Thread safety
    -------------
    ``FusionLayer``'s own lock protects only the ``_channels`` dict
    (channel creation / lookup).  Individual buffer reads and writes are
    serialised by each :class:`~cyberwave.data.ring_buffer.TimeIndexedRingBuffer`'s
    internal lock, so the outer lock is released before calling into the
    buffer.  This keeps concurrent ingestion on different channels fully
    parallel while still preventing dict mutation races.
    """

    def __init__(
        self,
        default_capacity: int = 1000,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._default_capacity = default_capacity
        self._clock = clock or time.time
        self._channels: dict[str, ChannelBuffer] = {}
        self._channel_capacities: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def clock(self) -> Callable[[], float]:
        """The clock function used for default timestamps."""
        return self._clock

    def configure_channel(self, channel: str, capacity: int) -> None:
        """Set a custom buffer depth for *channel*.

        Raises
        ------
        ValueError
            If samples have already been ingested on *channel*.
        """
        with self._lock:
            if channel in self._channels:
                raise ValueError(
                    f"Channel '{channel}' already has a buffer with "
                    f"{self._channels[channel].buffer.capacity} capacity. "
                    f"configure_channel() must be called before the first ingest()."
                )
            self._channel_capacities[channel] = capacity

    def _get_or_create(self, channel: str) -> ChannelBuffer:
        with self._lock:
            if channel not in self._channels:
                cap = self._channel_capacities.get(channel, self._default_capacity)
                self._channels[channel] = ChannelBuffer(channel, cap)
            return self._channels[channel]

    def ingest(self, channel: str, ts: float, value: Any) -> None:
        """Feed a decoded sample into the channel's ring buffer."""
        buf = self._get_or_create(channel)
        buf.ingest(ts, value)

    def at(
        self,
        channel: str,
        *,
        t: float | None = None,
        interpolation: str = "linear",
    ) -> Any | None:
        """Interpolated point read.

        Returns ``None`` for channels that have not received any samples
        via :meth:`ingest` (unknown channels).  This is the same value
        returned for an empty buffer, so callers can treat both cases
        uniformly.

        Parameters
        ----------
        channel:
            Channel name (e.g. ``"joint_states"``).
        t:
            Target timestamp.  Defaults to the clock provided at
            construction (``time.time()`` unless overridden).
        interpolation:
            ``"linear"`` (default), ``"slerp"`` (quaternions),
            ``"nearest"`` (closest sample), or ``"none"`` (exact match only).

        Raises
        ------
        ValueError
            If *interpolation* is not one of the recognised strategies.
        """
        if interpolation not in VALID_INTERPOLATIONS:
            raise ValueError(
                f"Invalid interpolation '{interpolation}'. "
                f"Must be one of: {', '.join(repr(i) for i in VALID_INTERPOLATIONS)}"
            )

        if t is None:
            t = self._clock()

        # The outer lock is released before calling into the buffer; each
        # TimeIndexedRingBuffer serialises its own reads/writes internally.
        with self._lock:
            buf = self._channels.get(channel)

        if buf is None:
            return None

        return buf.at(t, interpolation)

    def window(
        self,
        channel: str,
        *,
        from_t: float | None = None,
        to_t: float | None = None,
        duration_ms: float | None = None,
    ) -> WindowResult:
        """Time-range query.

        Returns an empty :class:`WindowResult` for channels that have not
        received any samples via :meth:`ingest` (unknown channels),
        consistent with ``at()`` returning ``None``.

        Parameters
        ----------
        channel:
            Channel name.
        from_t, to_t:
            Explicit time bounds (inclusive).
        duration_ms:
            Trailing window length in milliseconds.  Cannot be combined
            with ``from_t`` / ``to_t``.  Uses the *clock* provided at
            construction for the current time.

        Returns
        -------
        WindowResult
            Chronologically ordered samples within the range.

        Raises
        ------
        ValueError
            If the time-range arguments are invalid (neither
            ``(from_t, to_t)`` nor ``duration_ms`` given, or both).
            This is a parameter error, not a channel-existence error.
        """
        if duration_ms is not None and (from_t is not None or to_t is not None):
            raise ValueError("Specify either (from_t, to_t) or duration_ms, not both.")

        if duration_ms is not None:
            now = self._clock()
            to_t = now
            from_t = now - duration_ms / 1000.0
        elif from_t is None or to_t is None:
            raise ValueError("Must provide either (from_t and to_t) or duration_ms.")

        with self._lock:
            buf = self._channels.get(channel)

        if buf is None:
            return WindowResult(samples=[], from_t=from_t, to_t=to_t)

        return buf.window(from_t=from_t, to_t=to_t)

    @property
    def channels(self) -> list[str]:
        """List of channels with active buffers."""
        with self._lock:
            return list(self._channels.keys())

    def clear(self, channel: str | None = None) -> None:
        """Clear one or all channel buffers."""
        with self._lock:
            if channel is not None:
                buf = self._channels.get(channel)
                if buf is not None:
                    buf.buffer.clear()
            else:
                for buf in self._channels.values():
                    buf.buffer.clear()
