"""Hold a WebRTC offer until the broker has acked the answer subscriptions.

The SFU answers within milliseconds. An answer published before the broker
applies this client's SUBSCRIBE goes to nobody, and the stream waits out the
answer timeout before it retries.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

ANSWER_SUBSCRIPTION_TIMEOUT_S = 5.0


def answer_topics(client: Any, twin_uuid: str) -> list[str]:
    """The answer and candidate topics a WebRTC offer's reply arrives on."""
    prefix = client.topic_prefix
    return [
        f"{prefix}cyberwave/twin/{twin_uuid}/webrtc-answer",
        f"{prefix}cyberwave/twin/{twin_uuid}/webrtc-candidate",
    ]


async def wait_for_subscriptions(
    client: Any,
    topics: list[str],
    timeout: float | None = None,
    *,
    fallback_s: float = 0.0,
) -> None:
    """Wait until ``client.is_subscribed`` reports every topic, or *timeout*
    (default ``ANSWER_SUBSCRIPTION_TIMEOUT_S``).

    A client that cannot report SUBACKs waits *fallback_s* instead (default:
    no wait). A broker that never acks gets the offer after *timeout*.
    """
    is_subscribed = getattr(client, "is_subscribed", None)
    if not callable(is_subscribed):
        if fallback_s > 0:
            await asyncio.sleep(fallback_s)
        return
    if timeout is None:
        timeout = ANSWER_SUBSCRIPTION_TIMEOUT_S
    deadline = time.monotonic() + timeout
    while not all(is_subscribed(topic) for topic in topics):
        if time.monotonic() > deadline:
            logger.warning(
                "No SUBACK for %s after %.1fs; sending the offer anyway",
                ", ".join(topics),
                timeout,
            )
            return
        await asyncio.sleep(0.02)
