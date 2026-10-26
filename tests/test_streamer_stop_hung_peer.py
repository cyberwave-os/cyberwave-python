"""A streamer's stop must finish when the remote peer is gone and close() hangs.

``test_peer_close.py`` pins the helper; these pin that the streamers' own stop
paths go through it instead of awaiting ``pc.close()`` bare.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("av", reason="pyav not installed")
pytest.importorskip("aiortc", reason="aiortc not installed")

from cyberwave.consumers.video import IncomingVideoStream  # noqa: E402
from cyberwave.sensor import base_video, peer_close  # noqa: E402
from cyberwave.sensor.audio_microphone import (  # noqa: E402
    MicrophoneAudioStreamer as AudioMicrophoneStreamer,
)
from cyberwave.sensor.av_streamer import MultimediaStreamer  # noqa: E402
from cyberwave.sensor.microphone import MicrophoneAudioStreamer  # noqa: E402

_TIMEOUT_S = 0.2
# The shipped close cap, captured before the fixture below shortens it. Only a
# test at this value shows that an outer stop deadline outlasts the close: any
# fixed outer deadline wins against a shortened one.
_REAL_TIMEOUT_S = peer_close.PEER_CONNECTION_CLOSE_TIMEOUT_S


class _HungPeerConnection:
    """close() never returns, as when the relay or remote end has vanished."""

    connectionState = "connected"
    iceConnectionState = "completed"

    def __init__(self) -> None:
        self.close_called = False

    async def close(self) -> None:
        self.close_called = True
        await asyncio.Event().wait()


@pytest.fixture(autouse=True)
def _short_close_timeout(monkeypatch):
    monkeypatch.setattr(peer_close, "PEER_CONNECTION_CLOSE_TIMEOUT_S", _TIMEOUT_S)


def _multimedia_streamer() -> MultimediaStreamer:
    return MultimediaStreamer(
        client=MagicMock(topic_prefix=""),
        create_video_track=lambda: MagicMock(),
        create_audio_track=lambda: MagicMock(),
        twin_uuid="twin-av",
        enable_health_check=False,
    )


async def _bounded(coro, timeout: float = 5.0) -> float:
    started = time.monotonic()
    # The outer wait_for is the test's own guard: without the fix this hangs.
    await asyncio.wait_for(coro, timeout=timeout)
    return time.monotonic() - started


@pytest.mark.asyncio
async def test_multimedia_stop_abandons_a_hung_close():
    streamer = _multimedia_streamer()
    pc = _HungPeerConnection()
    streamer.pc = pc

    elapsed = await _bounded(streamer.stop())

    assert pc.close_called
    assert streamer.pc is None
    assert elapsed < _TIMEOUT_S + 1.0


@pytest.mark.asyncio
async def test_multimedia_failed_signaling_teardown_abandons_a_hung_close():
    streamer = _multimedia_streamer()
    pc = _HungPeerConnection()
    streamer.pc = pc

    elapsed = await _bounded(streamer._teardown_tracks_and_pc())

    assert pc.close_called
    assert streamer.pc is None
    assert elapsed < _TIMEOUT_S + 1.0


@pytest.mark.asyncio
async def test_audio_microphone_stop_abandons_a_hung_close():
    streamer = AudioMicrophoneStreamer(
        MagicMock(topic_prefix=""), lambda: None, twin_uuid="twin-mic"
    )
    pc = _HungPeerConnection()
    streamer.pc = pc

    elapsed = await _bounded(streamer.stop())

    assert pc.close_called
    assert streamer.pc is None
    assert elapsed < _TIMEOUT_S + 1.0


@pytest.mark.asyncio
async def test_audio_microphone_peer_close_abandons_a_hung_close():
    streamer = AudioMicrophoneStreamer(
        MagicMock(topic_prefix=""), lambda: None, twin_uuid="twin-mic"
    )
    pc = _HungPeerConnection()
    streamer.pc = pc

    elapsed = await _bounded(streamer._close_peer_connection())

    assert pc.close_called
    assert streamer.pc is None
    assert elapsed < _TIMEOUT_S + 1.0


class _VideoStreamer(base_video.BaseVideoStreamer):
    def initialize_track(self):
        raise NotImplementedError


class _SignalingFailed(Exception):
    pass


def _fail_signaling_with_a_hung_pc(monkeypatch, streamer, pc) -> None:
    """Make start() build ``pc`` and then fail signaling, as when no answer comes."""

    async def setup() -> None:
        streamer.pc = pc

    async def fail() -> None:
        raise _SignalingFailed

    monkeypatch.setattr(streamer, "_setup_webrtc", setup)
    monkeypatch.setattr(streamer, "_perform_signaling", fail)


@pytest.mark.asyncio
async def test_base_video_failed_signaling_abandons_a_hung_close(monkeypatch):
    streamer = _VideoStreamer(
        client=MagicMock(topic_prefix=""),
        twin_uuid="twin-cam",
        enable_health_check=False,
    )
    pc = _HungPeerConnection()
    real_sleep = asyncio.sleep

    async def short_sleep(delay):
        # start() waits 2.5 s for the answer subscription; not under test here.
        await real_sleep(min(delay, 0.01))

    monkeypatch.setattr(base_video.asyncio, "sleep", short_sleep)
    monkeypatch.setattr(
        base_video, "apply_h264_hw_patch", lambda: SimpleNamespace(codec_name="libx264")
    )
    monkeypatch.setattr(streamer, "_subscribe_to_answer", MagicMock())
    _fail_signaling_with_a_hung_pc(monkeypatch, streamer, pc)

    started = time.monotonic()
    with pytest.raises(_SignalingFailed):
        await asyncio.wait_for(streamer.start(), timeout=5.0)

    assert pc.close_called
    assert streamer.pc is None
    assert time.monotonic() - started < _TIMEOUT_S + 1.0


@pytest.mark.asyncio
async def test_microphone_failed_signaling_abandons_a_hung_close(monkeypatch):
    streamer = MicrophoneAudioStreamer(
        MagicMock(topic_prefix=""),
        lambda: None,
        twin_uuid="twin-mic",
        enable_health_check=False,
    )
    pc = _HungPeerConnection()
    _fail_signaling_with_a_hung_pc(monkeypatch, streamer, pc)

    started = time.monotonic()
    with pytest.raises(_SignalingFailed):
        await asyncio.wait_for(streamer._start_webrtc(), timeout=5.0)

    assert pc.close_called
    assert streamer.pc is None
    assert time.monotonic() - started < _TIMEOUT_S + 1.0


def _messages(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def _helper_gave_up(caplog) -> bool:
    return any("did not close within" in m for m in _messages(caplog))


@pytest.mark.parametrize("close_timeout_s", [_TIMEOUT_S, _REAL_TIMEOUT_S])
def test_video_consumer_stop_abandons_a_hung_close(
    monkeypatch, caplog, close_timeout_s
):
    monkeypatch.setattr(peer_close, "PEER_CONNECTION_CLOSE_TIMEOUT_S", close_timeout_s)
    caplog.set_level(logging.DEBUG, logger="cyberwave")
    stream = IncomingVideoStream(MagicMock(topic_prefix=""), "twin-view")
    # The consumer's own loop thread, as start() runs it, minus the signaling.
    stream._thread = threading.Thread(target=stream._run_loop, daemon=True)
    stream._thread.start()
    assert stream._loop_ready.wait(timeout=5.0)
    pc = _HungPeerConnection()
    stream._pc = pc
    drained = threading.Event()
    # The loop holds tasks only weakly; keep this one alive until the drain.
    gate = asyncio.Event()

    async def pending_aiortc_task() -> None:
        try:
            await gate.wait()
        except asyncio.CancelledError:
            drained.set()
            raise

    pending = asyncio.run_coroutine_threadsafe(pending_aiortc_task(), stream._loop)

    started = time.monotonic()
    stream.stop()
    elapsed = time.monotonic() - started

    assert pc.close_called
    # The bounded close gave up, and stop()'s own wait did not cut _teardown
    # off before that: the pending-task drain after the close ran.
    assert _helper_gave_up(caplog)
    assert "teardown error (ignored)" not in _messages(caplog)
    assert drained.is_set()
    assert pending.cancelled()
    assert elapsed < close_timeout_s + 1.0
    stream._thread.join(timeout=5.0)
    assert not stream._thread.is_alive()


class _CleanupOutcome:
    """Records whether a run task's cleanup finished or was cancelled midway."""

    def __init__(self, monkeypatch, streamer) -> None:
        self.outcome: list[str] = []
        real_cleanup = streamer._cleanup_run

        async def cleanup() -> None:
            try:
                await real_cleanup()
            except asyncio.CancelledError:
                self.outcome.append("cancelled")
                raise
            self.outcome.append("finished")

        monkeypatch.setattr(streamer, "_cleanup_run", cleanup)


def _run_task_streamers():
    def multimedia():
        streamer = _multimedia_streamer()
        streamer.video_track = MagicMock()
        streamer.audio_track = MagicMock()
        return streamer, [streamer.video_track, streamer.audio_track]

    def audio_microphone():
        streamer = AudioMicrophoneStreamer(
            MagicMock(topic_prefix=""), lambda: None, twin_uuid="twin-mic"
        )
        streamer.streamer = MagicMock()
        return streamer, [streamer.streamer]

    def microphone():
        streamer = MicrophoneAudioStreamer(
            MagicMock(topic_prefix=""),
            lambda: None,
            twin_uuid="twin-mic",
            enable_health_check=False,
        )
        streamer.streamer = MagicMock()
        return streamer, [streamer.streamer]

    return [
        pytest.param(multimedia, id="multimedia"),
        pytest.param(audio_microphone, id="audio_microphone"),
        pytest.param(microphone, id="microphone"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("make_streamer", _run_task_streamers())
async def test_stop_outlasts_the_run_tasks_bounded_close(
    monkeypatch, caplog, make_streamer
):
    """stop() must let the run task's own cleanup finish its bounded close.

    Run at the shipped close cap: stop() used to give the run task a fixed
    5 s, which the 5 s close inside its cleanup can never beat.
    """
    monkeypatch.setattr(peer_close, "PEER_CONNECTION_CLOSE_TIMEOUT_S", _REAL_TIMEOUT_S)
    caplog.set_level(logging.WARNING, logger="cyberwave")
    streamer, tracks = make_streamer()
    cleanup = _CleanupOutcome(monkeypatch, streamer)
    pc = _HungPeerConnection()
    streamer.pc = pc
    await streamer.start()
    await asyncio.sleep(0.1)  # Let the run loop come up with the hung peer.

    elapsed = await _bounded(streamer.stop(), timeout=_REAL_TIMEOUT_S + 5.0)

    assert pc.close_called
    assert cleanup.outcome == ["finished"]
    assert _helper_gave_up(caplog)
    for track in tracks:
        track.close.assert_called_once()
    assert streamer.pc is None
    assert elapsed < peer_close.close_budget_s(extra=1.5)
