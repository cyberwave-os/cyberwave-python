"""Data-layer helpers for Zenoh frame reads and wire-key utilities.

Used by :meth:`TwinCameraHandle.get_frame` (``source='zenoh'``) and diagnostics.
Drivers publish with Zenoh **put**; ``session.get()`` often returns nothing while
subscriptions work — this module subscribes on ``frames/**`` (same keys as
``@cw.on_frame``) and serves blocking reads from the live stream.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .backend import Sample
from .exceptions import ChannelError
from .keys import build_key, build_wildcard, parse_key

logger = logging.getLogger(__name__)

_SUBSCRIBER_WARMUP_S = 0.15


def sensor_name_from_frame_channel(channel: str, *, default: str = "default") -> str:
    """Map a wire key or hook channel to the ``frames/<sensor>`` segment."""
    if not channel or "/data/" not in channel:
        return default
    try:
        parsed = parse_key(channel)
    except ChannelError:
        return default
    if parsed.sensor_name:
        return parsed.sensor_name
    if "/" in parsed.channel:
        return parsed.channel.split("/", 1)[1]
    return default


def frame_subscribe_listen_keys(
    twin_uuid: str,
    sensor_names: tuple[str, ...],
    *,
    key_prefix: str = "cw",
) -> list[str]:
    """Zenoh keys for camera frames — wildcard plus per-sensor exact keys."""
    keys: list[str] = [build_wildcard(twin_uuid, "frames", prefix=key_prefix)]
    for sensor in sensor_names:
        keys.append(build_key(twin_uuid, "frames", sensor, prefix=key_prefix))
        compound = f"frames/{sensor}"
        compound_key = build_key(twin_uuid, compound, None, prefix=key_prefix)
        if compound_key not in keys:
            keys.append(compound_key)
    return keys


class _TwinFrameSubscribeCache:
    """Per-twin subscribers on ``frames`` with per-sensor latest decoded frames."""

    def __init__(
        self,
        backend: Any,
        twin_uuid: str,
        *,
        sensor_names: tuple[str, ...] = ("default",),
        key_prefix: str = "cw",
    ) -> None:
        self._lock = threading.Lock()
        self._latest_by_sensor: dict[str, tuple[Any, float]] = {}
        self._updated = threading.Event()
        self._closed = False
        self._subscriptions: list[Any] = []

        seen: set[str] = set()
        for key in frame_subscribe_listen_keys(
            twin_uuid, sensor_names, key_prefix=key_prefix
        ):
            if key in seen:
                continue
            seen.add(key)
            try:
                sub = backend.subscribe(
                    key,
                    self._on_sample,
                    policy="latest",
                )
                self._subscriptions.append(sub)
                logger.debug("Frame subscribe cache listening on %s", key)
            except Exception:
                logger.warning(
                    "Frame subscribe cache could not subscribe to %s",
                    key,
                    exc_info=True,
                )

        time.sleep(_SUBSCRIBER_WARMUP_S)

    def _on_sample(self, sample: Sample) -> None:
        # JPEG + SDK wire decode lives in workers.decode (shared with hooks).
        from cyberwave.workers.decode import decode_sample_payload

        try:
            decoded, ts = decode_sample_payload(sample, content_hint="numpy")
        except Exception:
            logger.debug(
                "Frame subscribe cache dropped sample on %s",
                getattr(sample, "channel", "?"),
                exc_info=True,
            )
            return
        sensor = sensor_name_from_frame_channel(getattr(sample, "channel", "") or "")
        with self._lock:
            self._latest_by_sensor[sensor] = (decoded, ts)
        self._updated.set()

    def fetch(
        self,
        *,
        sensor_name: str,
        timeout_s: float,
        max_age_ms: float | None,
    ) -> Any | None:
        deadline = time.monotonic() + max(timeout_s, 0.0)
        preferred = (sensor_name, "default")

        while time.monotonic() < deadline:
            frame = self._take_matching(preferred, max_age_ms=max_age_ms)
            if frame is not None:
                return frame
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if self._updated.wait(timeout=remaining):
                self._updated.clear()
        return None

    def _take_matching(
        self,
        sensors: tuple[str, ...],
        *,
        max_age_ms: float | None,
    ) -> Any | None:
        now = time.time()
        with self._lock:
            for sensor in sensors:
                entry = self._latest_by_sensor.get(sensor)
                if entry is None:
                    continue
                frame, ts = entry
                if max_age_ms is not None and (now - ts) * 1000.0 > max_age_ms:
                    continue
                return frame
            for frame, ts in self._latest_by_sensor.values():
                if max_age_ms is not None and (now - ts) * 1000.0 > max_age_ms:
                    continue
                return frame
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for sub in self._subscriptions:
            try:
                sub.close()
            except Exception:
                pass


_frame_caches_lock = threading.Lock()
_frame_caches: dict[tuple[str, int], _TwinFrameSubscribeCache] = {}


def fetch_twin_frame(
    backend: Any,
    twin_uuid: str,
    *,
    sensor_name: str = "default",
    timeout_s: float = 1.0,
    max_age_ms: float | None = None,
    key_prefix: str = "cw",
) -> Any | None:
    """Blocking read of the next camera frame via Zenoh subscribe (not ``get()``)."""
    cache_key = (twin_uuid, id(backend))
    sensors = tuple(dict.fromkeys((sensor_name, "default")))
    with _frame_caches_lock:
        cache = _frame_caches.get(cache_key)
        if cache is None:
            cache = _TwinFrameSubscribeCache(
                backend,
                twin_uuid,
                sensor_names=sensors,
                key_prefix=key_prefix,
            )
            _frame_caches[cache_key] = cache
    frame = cache.fetch(
        sensor_name=sensor_name,
        timeout_s=timeout_s,
        max_age_ms=max_age_ms,
    )
    if frame is None:
        stats: dict[str, Any] = {}
        stats_fn = getattr(backend, "stats", None)
        if stats_fn is not None:
            stats = stats_fn()
        logger.warning(
            "Zenoh frame fetch timed out for twin %s (sensor=%r, timeout=%.1fs). "
            "recv=%s. Set ZENOH_CONNECT to the same router as the camera driver.",
            twin_uuid,
            sensor_name,
            timeout_s,
            stats.get("recv"),
        )
    return frame


def close_frame_subscribe_caches_for_backend(backend: Any) -> None:
    """Drop frame subscribe caches for *backend* (``Cyberwave.disconnect``)."""
    backend_id = id(backend)
    with _frame_caches_lock:
        doomed = [key for key in _frame_caches if key[1] == backend_id]
        for key in doomed:
            cache = _frame_caches.pop(key, None)
            if cache is not None:
                cache.close()
