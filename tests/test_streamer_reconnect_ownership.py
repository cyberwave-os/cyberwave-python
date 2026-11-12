"""Who retries a WebRTC stream, for every streamer that runs its own monitor.

The run loop retries only a first connect that failed. Once the stream has
connected, by the run loop or by a start command, every retry belongs to the
monitor: a second offer from the run loop would overwrite (and leak) the peer
connection a reconnect is building, or undo a stop command. A failed reconnect
leaves pc None, so the monitor tracks it in ``_reconnect_pending`` instead of
waiting for a pc that never comes. Retries never give up on their own and wait
while MQTT is down, and a cancelled reconnect parks the monitor rather than
ending it, so a later start command is watched again.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

pytest.importorskip("av", reason="pyav not installed")
pytest.importorskip("aiortc", reason="aiortc not installed")

from cyberwave.sensor import (  # noqa: E402
    audio_microphone,
    av_streamer,
    base_video,
    microphone,
)


class _Camera(base_video.BaseVideoStreamer):
    def initialize_track(self):  # pragma: no cover - start() is faked
        raise NotImplementedError


def _client() -> MagicMock:
    client = MagicMock()
    client.topic_prefix = ""
    client.client_id = "test-client"
    return client


def _silence() -> bytes | None:
    return None


@dataclass
class _Kind:
    name: str
    module: ModuleType
    make: Callable[[], Any]
    # The method that negotiates a fresh peer connection, and the one the
    # monitor tears the old one down with.
    start_attr: str
    close_attr: str
    run: Callable[[Any, asyncio.Event], Any]
    # The microphone's stop command only stops recording; the stream stays.
    stop_command_stops_stream: bool = True


KINDS = [
    _Kind(
        "camera",
        base_video,
        lambda: _Camera(
            client=_client(), twin_uuid="twin-1", enable_health_check=False
        ),
        "start",
        "stop",
        lambda s, stop: s.run_with_auto_reconnect(
            stop_event=stop, subscribe_to_commands=False
        ),
    ),
    _Kind(
        "microphone",
        microphone,
        lambda: microphone.MicrophoneAudioStreamer(
            _client(),
            get_audio=_silence,
            twin_uuid="twin-1",
            mic_name="mic",
            enable_health_check=False,
        ),
        "_start_webrtc",
        "_close_peer_connection",
        lambda s, stop: s.run_with_auto_reconnect(
            stop_event=stop, subscribe_to_commands=False
        ),
        stop_command_stops_stream=False,
    ),
    _Kind(
        "multimedia",
        av_streamer,
        lambda: av_streamer.MultimediaStreamer(
            _client(),
            create_video_track=Mock(),
            create_audio_track=Mock(),
            twin_uuid="twin-1",
            camera_name="cam",
            mic_name="mic",
            enable_health_check=False,
        ),
        "_start_webrtc",
        "_close_peer_connection",
        lambda s, stop: s._run_with_auto_reconnect(stop_event=stop),
    ),
    _Kind(
        "audio_microphone",
        audio_microphone,
        lambda: audio_microphone.MicrophoneAudioStreamer(
            _client(), get_audio=_silence, twin_uuid="twin-1", sensor_name="mic"
        ),
        "_start_webrtc",
        "_close_peer_connection",
        lambda s, stop: s.run_with_auto_reconnect(stop_event=stop),
    ),
]
STOPPABLE = [k for k in KINDS if k.stop_command_stops_stream]


def _ids(kinds: list[_Kind]) -> list[str]:
    return [k.name for k in kinds]


def _connected() -> SimpleNamespace:
    return SimpleNamespace(connectionState="connected", iceConnectionState="completed")


def _failed() -> SimpleNamespace:
    return SimpleNamespace(connectionState="failed", iceConnectionState="failed")


class _Harness:
    """A streamer with faked negotiation, a 20 s step clock and instant sleeps."""

    def __init__(self, kind: _Kind, monkeypatch: pytest.MonkeyPatch) -> None:
        self.kind = kind
        self.streamer = kind.make()
        self.starts: list[int] = []
        # Starts numbered below this fail, like an SFU that never answers.
        self.fail_below = 0
        self.real_sleep = asyncio.sleep

        async def start(*_args: Any, **_kwargs: Any) -> None:
            self.starts.append(len(self.starts))
            if len(self.starts) <= self.fail_below:
                raise TimeoutError("Timeout waiting for WebRTC answer")
            self.streamer.pc = _connected()

        async def close(*_args: Any, **_kwargs: Any) -> None:
            self.streamer.pc = None

        monkeypatch.setattr(self.streamer, kind.start_attr, start)
        monkeypatch.setattr(self.streamer, kind.close_attr, close)
        # Every reading is 20 s after the last, so the run loop's 15 s first
        # retry delay and its backoff have always elapsed by the next pass.
        clock = itertools.count(0.0, 20.0)
        monkeypatch.setattr(
            kind.module,
            "time",
            SimpleNamespace(monotonic=lambda: next(clock), time=time.time),
        )

        async def yield_once(_delay: float) -> None:
            await self.real_sleep(0)

        monkeypatch.setattr(kind.module.asyncio, "sleep", yield_once)

    async def spin(self, passes: int = 200) -> None:
        for _ in range(passes):
            await self.real_sleep(0)

    def start_monitor(self) -> tuple[asyncio.Task, asyncio.Event]:
        """Run the monitor on a stream whose WebRTC link has just dropped."""
        self.streamer.pc = _failed()
        self.streamer._should_reconnect = True
        self.streamer._is_running = True
        stop = asyncio.Event()
        return asyncio.create_task(self.streamer._monitor_connection(stop)), stop

    def start_run_loop(self) -> tuple[asyncio.Task, asyncio.Event]:
        stop = asyncio.Event()
        return asyncio.create_task(self.kind.run(self.streamer, stop)), stop


async def _finish(task: asyncio.Task, stop: asyncio.Event) -> None:
    stop.set()
    await asyncio.wait_for(task, timeout=5.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS, ids=_ids(KINDS))
async def test_the_monitor_retries_after_a_failed_reconnect(kind, monkeypatch):
    h = _Harness(kind, monkeypatch)
    h.fail_below = 2
    task, stop = h.start_monitor()

    await h.spin()

    # Two failed attempts, then the third reconnects; the monitor does not
    # stall on the pc a failed attempt leaves as None.
    assert h.starts == [0, 1, 2]
    assert h.streamer.pc is not None
    assert h.streamer._reconnect_pending is False
    await _finish(task, stop)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS, ids=_ids(KINDS))
async def test_the_monitor_keeps_retrying_past_the_fast_attempts(kind, monkeypatch):
    # An uplink outage outlasts any fixed number of attempts; a stream that
    # parked after ten would stay dead until the process restarts.
    h = _Harness(kind, monkeypatch)
    h.fail_below = 14
    task, stop = h.start_monitor()

    await h.spin(4000)

    assert len(h.starts) == 15
    assert h.streamer.pc is not None
    assert h.streamer._should_reconnect is True
    assert h.streamer._reconnect_pending is False
    await _finish(task, stop)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS, ids=_ids(KINDS))
async def test_the_monitor_holds_off_while_mqtt_is_down(kind, monkeypatch):
    h = _Harness(kind, monkeypatch)
    h.streamer.client.connected = False
    task, stop = h.start_monitor()

    await h.spin()
    # No offer can be answered without the broker, so none is sent.
    assert h.starts == []

    h.streamer.client.connected = True
    await h.spin()

    assert h.starts == [0]
    assert h.streamer.pc is not None
    await _finish(task, stop)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", STOPPABLE, ids=_ids(STOPPABLE))
async def test_a_stop_command_between_reconnect_attempts_ends_them(kind, monkeypatch):
    h = _Harness(kind, monkeypatch)
    h.fail_below = 1000
    task, stop = h.start_monitor()
    # A few failed attempts.
    await h.spin(15)
    assert h.streamer.pc is None
    assert h.streamer._reconnect_pending is True

    # pc is None here, which a stop command used to read as "not running".
    await h.streamer._handle_stop_command()
    await h.spin(10)
    attempts = len(h.starts)
    await h.spin()

    assert attempts > 0
    assert len(h.starts) == attempts
    assert h.streamer._should_reconnect is False
    assert h.streamer._reconnect_pending is False
    await _finish(task, stop)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS, ids=_ids(KINDS))
async def test_a_start_command_after_a_failed_first_connect_ends_the_run_loop_retry(
    kind, monkeypatch
):
    h = _Harness(kind, monkeypatch)
    h.fail_below = 1000
    task, stop = h.start_run_loop()
    await h.spin(40)
    assert len(h.starts) >= 2  # the first connect and at least one retry
    assert h.streamer.pc is None

    h.fail_below = 0
    await h.streamer._handle_start_command()
    assert h.streamer.pc is not None
    started = len(h.starts)

    # What a monitor reconnect (stop, wait, start) leaves behind: the run loop
    # must not send a second offer over the one the monitor is building.
    h.streamer.pc = None
    await h.spin()

    assert len(h.starts) == started
    await _finish(task, stop)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", STOPPABLE, ids=_ids(STOPPABLE))
async def test_a_stop_command_ends_the_first_connect_retry(kind, monkeypatch):
    h = _Harness(kind, monkeypatch)
    h.fail_below = 1000
    task, stop = h.start_run_loop()
    await h.spin(40)
    assert h.streamer.pc is None

    await h.streamer._handle_stop_command()
    stopped = len(h.starts)
    h.fail_below = 0
    await h.spin()

    assert len(h.starts) == stopped
    assert h.streamer.pc is None
    await _finish(task, stop)


# ---------------------------------------------------------------------------
# The multimedia and audio_microphone offers wait for the answer SUBACKs too.
# ---------------------------------------------------------------------------

ANSWER = "cyberwave/twin/twin-1/webrtc-answer"
CANDIDATE = "cyberwave/twin/twin-1/webrtc-candidate"
OFFER_KINDS = [k for k in KINDS if k.name in ("multimedia", "audio_microphone")]


def _signaling_streamer(kind: _Kind, client: MagicMock) -> Any:
    streamer = kind.make()
    streamer.client = client
    pc = MagicMock()
    pc.createOffer = AsyncMock(return_value=MagicMock())
    pc.setLocalDescription = AsyncMock()
    pc.iceGatheringState = "complete"
    pc.localDescription.type = "offer"
    pc.localDescription.sdp = (
        "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=rtpmap:111 opus/48000/2\r\n"
    )
    streamer.pc = pc
    return streamer


def _offers(client: MagicMock) -> list[Any]:
    return [
        c for c in client.publish.call_args_list if c.args[0].endswith("/webrtc-offer")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", OFFER_KINDS, ids=_ids(OFFER_KINDS))
async def test_the_offer_goes_out_only_after_both_answer_subacks(kind, monkeypatch):
    client = _client()
    acked: set[str] = set()
    client.is_subscribed = lambda topic: topic in acked
    streamer = _signaling_streamer(kind, client)
    monkeypatch.setattr(streamer, "_wait_for_answer", AsyncMock())

    signaling = asyncio.create_task(streamer._perform_signaling())
    await asyncio.sleep(0.1)
    acked.add(ANSWER)
    await asyncio.sleep(0.1)
    assert _offers(client) == []

    acked.add(CANDIDATE)
    await asyncio.wait_for(signaling, timeout=1.0)
    assert len(_offers(client)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", OFFER_KINDS, ids=_ids(OFFER_KINDS))
async def test_a_broker_that_never_acks_still_gets_the_offer(kind, monkeypatch):
    from cyberwave.sensor import answer_subscription

    client = _client()
    client.is_subscribed = lambda _topic: False
    streamer = _signaling_streamer(kind, client)
    monkeypatch.setattr(streamer, "_wait_for_answer", AsyncMock())
    monkeypatch.setattr(answer_subscription, "ANSWER_SUBSCRIPTION_TIMEOUT_S", 0.05)

    await asyncio.wait_for(streamer._perform_signaling(), timeout=1.0)
    assert len(_offers(client)) == 1
