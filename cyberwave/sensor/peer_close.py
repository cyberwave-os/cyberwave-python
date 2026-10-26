"""Bounded WebRTC peer-connection close shared by the video and audio streamers.

``RTCPeerConnection.close()`` has no deadline of its own: it awaits the DTLS,
SCTP and ICE transports in turn, and an ICE transport whose TURN allocation or
consent check is stuck on an unreachable relay can keep it pending for as long
as aioice's STUN retransmits run (tens of seconds). A stop that awaits it bare
therefore inherits that worst case, and so does every caller that has a stop
budget of its own (the simulator's worker, an edge driver's SIGTERM handler).

Kept dependency-free so it can be imported without :mod:`aiortc` and tested in
isolation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Upper bound on one ``pc.close()``. A healthy close takes well under a second;
# past this the transports are abandoned and left to the garbage collector.
PEER_CONNECTION_CLOSE_TIMEOUT_S = 5.0

# Pause after a successful close so aioice's already-scheduled STUN retry
# callbacks fire against a live loop instead of a closed one. Skipped when the
# close timed out: nothing has settled then, and waiting longer would not help.
PEER_CONNECTION_SETTLE_S = 1.5

# Margin an outer wait keeps over the bounded close it contains.
_OUTER_WAIT_SLACK_S = 1.0


def close_budget_s(*, settle: float | None = None, extra: float = 0.0) -> float:
    """How long an outer wait around a :func:`close_peer_connection` must allow.

    That is the close's own timeout, its ``settle`` (default constant when
    ``None``), ``extra`` for whatever else the caller's teardown does around the
    close, and a second of slack. An outer deadline that is no longer than the
    inner one fires first and cuts the teardown off partway, so the close's own
    bound never applies. The constants are read at call time, so patching
    them moves the outer deadline together with the inner one.
    """
    if settle is None:
        settle = PEER_CONNECTION_SETTLE_S
    return PEER_CONNECTION_CLOSE_TIMEOUT_S + settle + extra + _OUTER_WAIT_SLACK_S


async def close_peer_connection(
    pc: Any,
    *,
    timeout: float | None = None,
    settle: float | None = None,
    error_log_level: int = logging.ERROR,
) -> bool:
    """Close ``pc`` within ``timeout`` seconds; return whether it closed cleanly.

    ``timeout`` and ``settle`` default to the module constants, read at call
    time. Never raises for a slow or failing close: the caller is tearing down
    and drops its reference either way. Cancellation of the caller still
    propagates. ``error_log_level`` is the level a failing close is logged at;
    teardown after a failed signaling, where the peer was never usable, passes
    ``logging.DEBUG``. A timed-out close is always a warning.
    """
    if timeout is None:
        timeout = PEER_CONNECTION_CLOSE_TIMEOUT_S
    if settle is None:
        settle = PEER_CONNECTION_SETTLE_S
    try:
        await asyncio.wait_for(pc.close(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "WebRTC peer connection did not close within %.1fs; abandoning it",
            timeout,
        )
        return False
    except Exception as error:
        logger.log(
            error_log_level,
            "Error closing peer connection: %s",
            error,
            exc_info=error_log_level <= logging.DEBUG,
        )
        return False
    if settle > 0:
        await asyncio.sleep(settle)
    return True
