"""A camera offer waits for the broker to ack its answer subscriptions.

The SFU answers within milliseconds. An answer published before the broker
applies the producer's ``webrtc-answer`` SUBSCRIBE goes to nobody, and the
stream waits out the answer timeout. A fixed 2.5 s pause stood in for the
SUBACK: it delayed every camera start and still lost the race on a loaded
broker.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("aiortc", reason="aiortc not installed")

from cyberwave.mqtt_client import CyberwaveMQTTClient  # noqa: E402
from cyberwave.sensor import base_video  # noqa: E402
from cyberwave.sensor.base_video import BaseVideoStreamer  # noqa: E402

ANSWER = "cyberwave/twin/twin-1/webrtc-answer"
CANDIDATE = "cyberwave/twin/twin-1/webrtc-candidate"


class _FakeMQTT:
    topic_prefix = ""
    client_id = "test-client"

    def __init__(self) -> None:
        self.acked: set[str] = set()

    def subscribe(self, *args: Any, **kwargs: Any) -> None:
        pass

    def publish(self, *args: Any, **kwargs: Any) -> None:
        pass

    def is_subscribed(self, topic: str) -> bool:
        return topic in self.acked


class _NoAckMQTT:
    """A client that cannot report SUBACKs."""

    topic_prefix = ""
    client_id = "test-client"

    def subscribe(self, *args: Any, **kwargs: Any) -> None:
        pass


class _Streamer(BaseVideoStreamer):
    def initialize_track(self):  # pragma: no cover - not exercised here
        raise NotImplementedError


def _streamer(client: Any) -> _Streamer:
    return _Streamer(
        client=client,
        twin_uuid="twin-1",
        camera_name="wrist",
        auto_reconnect=False,
        enable_health_check=False,
    )


@pytest.mark.asyncio
async def test_the_offer_waits_for_both_answer_topics_to_be_acked():
    client = _FakeMQTT()
    streamer = _streamer(client)

    waiting = asyncio.create_task(streamer._wait_for_answer_subscription())
    await asyncio.sleep(0.05)
    assert not waiting.done()

    client.acked.add(ANSWER)
    await asyncio.sleep(0.05)
    assert not waiting.done()  # candidates trickle in on their own topic

    client.acked.add(CANDIDATE)
    await asyncio.wait_for(waiting, timeout=1.0)


@pytest.mark.asyncio
async def test_an_acked_subscription_costs_no_pause():
    client = _FakeMQTT()
    client.acked.update({ANSWER, CANDIDATE})
    streamer = _streamer(client)

    started = time.monotonic()
    await streamer._wait_for_answer_subscription()

    assert time.monotonic() - started < 0.5


@pytest.mark.asyncio
async def test_a_broker_that_never_acks_still_gets_the_offer():
    streamer = _streamer(_FakeMQTT())

    await asyncio.wait_for(
        streamer._wait_for_answer_subscription(timeout=0.05), timeout=1.0
    )


@pytest.mark.asyncio
async def test_a_client_without_suback_tracking_keeps_the_fixed_pause():
    streamer = _streamer(_NoAckMQTT())
    pauses: list[float] = []

    async def record(seconds: float) -> None:
        pauses.append(seconds)

    with patch.object(base_video.asyncio, "sleep", record):
        await streamer._wait_for_answer_subscription()

    assert pauses == [base_video.ANSWER_SUBSCRIPTION_FALLBACK_S]


@pytest.mark.asyncio
async def test_start_subscribes_and_waits_before_building_the_offer():
    streamer = _streamer(_FakeMQTT())
    order: list[str] = []

    async def wait() -> None:
        order.append("wait")

    async def setup() -> None:
        order.append("setup")
        raise RuntimeError("stop here")

    with (
        patch.object(
            base_video, "apply_h264_hw_patch", return_value=MagicMock(codec_name=None)
        ),
        patch.object(
            streamer, "_subscribe_to_answer", lambda: order.append("subscribe")
        ),
        patch.object(streamer, "_wait_for_answer_subscription", wait),
        patch.object(streamer, "_setup_webrtc", setup),
        pytest.raises(RuntimeError, match="stop here"),
    ):
        await streamer.start()

    assert order == ["subscribe", "wait", "setup"]


def test_the_sdk_mqtt_wrapper_reports_acks():
    """``cw.mqtt`` is what streamers are handed; it must expose the ack state."""
    wrapper = CyberwaveMQTTClient.__new__(CyberwaveMQTTClient)
    wrapper._client = MagicMock()
    wrapper._client.is_subscribed.side_effect = lambda topic: topic == ANSWER

    assert wrapper.is_subscribed(ANSWER)
    assert not wrapper.is_subscribed(CANDIDATE)
