"""Rate limiting for high-frequency ROS/device streams forwarded to Cyber MQTT/Zenoh."""

from __future__ import annotations

import time


class StreamPublishRateLimiter:
    """Per-stream monotonic throttle (``max_hz`` cap, latest-wins on acquire)."""

    def __init__(self) -> None:
        self._last_at: dict[str, float] = {}

    def acquire(self, stream_key: str, *, max_hz: float) -> bool:
        """Return ``True`` when at least ``1/max_hz`` seconds elapsed since last acquire."""
        if max_hz <= 0:
            return True
        now = time.monotonic()
        interval = 1.0 / max_hz
        last = self._last_at.get(stream_key, 0.0)
        if now - last < interval:
            return False
        self._last_at[stream_key] = now
        return True
