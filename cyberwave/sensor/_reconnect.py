"""Reconnect pacing shared by the WebRTC streamers.

A streamer on an edge device must outlive any uplink outage, so it never stops
retrying on its own: after ``FAST_RECONNECT_ATTEMPTS`` failures it keeps going
every ``SLOW_RECONNECT_DELAY`` seconds. Only an explicit stop ends the retries.
"""

from __future__ import annotations

from typing import Any

RECONNECT_BASE_DELAY = 2.0
RECONNECT_MAX_DELAY = 30.0
FAST_RECONNECT_ATTEMPTS = 10
SLOW_RECONNECT_DELAY = 60.0


def reconnect_backoff(attempt: int) -> float:
    """Seconds to wait after the ``attempt``-th consecutive failure."""
    if attempt >= FAST_RECONNECT_ATTEMPTS:
        return SLOW_RECONNECT_DELAY
    return min(RECONNECT_BASE_DELAY * (2**attempt), RECONNECT_MAX_DELAY)


def signaling_ready(client: Any) -> bool:
    """Whether an offer published now can reach the backend.

    While MQTT is down an offer goes nowhere and the attempt only waits out
    its answer timeout, so the monitor holds off until the broker is back.
    A client without a ``connected`` flag is assumed ready.
    """
    return bool(getattr(client, "connected", True))
