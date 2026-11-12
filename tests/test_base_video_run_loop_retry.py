"""The run loop retries only a failed first connect; later drops belong to the monitor."""

import asyncio
import itertools
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("aiortc")

from cyberwave.sensor import base_video  # noqa: E402


class _Streamer(base_video.BaseVideoStreamer):
    def initialize_track(self):
        raise NotImplementedError


def _streamer(monkeypatch, *, first_start_fails: bool):
    streamer = _Streamer(
        client=Mock(), twin_uuid="camera-twin", enable_health_check=False
    )
    starts: list[int] = []

    async def start(twin_uuid=None):
        starts.append(len(starts))
        if first_start_fails and len(starts) == 1:
            raise TimeoutError("Timeout waiting for WebRTC answer")
        streamer.pc = SimpleNamespace(
            connectionState="connected", iceConnectionState="completed"
        )

    monkeypatch.setattr(streamer, "start", start)
    monkeypatch.setattr(streamer, "stop", AsyncMock())
    # Every reading is 20 s after the last, so the 15 s first-retry delay and
    # the 30 s backoff have always elapsed by the next loop pass.
    clock = itertools.count(0.0, 20.0)
    monkeypatch.setattr(
        base_video,
        "time",
        SimpleNamespace(monotonic=lambda: next(clock), time=time.time),
    )
    real_sleep = asyncio.sleep

    async def yield_once(_delay):
        await real_sleep(0)

    monkeypatch.setattr(base_video.asyncio, "sleep", yield_once)
    return streamer, starts, real_sleep


async def _run(streamer, real_sleep, between):
    stop = asyncio.Event()
    run = asyncio.create_task(
        streamer.run_with_auto_reconnect(stop_event=stop, subscribe_to_commands=False)
    )
    for _ in range(20):
        await real_sleep(0)
    between()
    for _ in range(20):
        await real_sleep(0)
    stop.set()
    await run


@pytest.mark.asyncio
async def test_a_dropped_stream_is_not_restarted_by_the_run_loop(monkeypatch):
    streamer, starts, real_sleep = _streamer(monkeypatch, first_start_fails=False)

    def drop():
        # What the monitor's reconnect (stop, wait, start) or a stop_video
        # command leaves behind. A second start() here would overwrite the
        # peer connection the reconnect is building, or undo the stop.
        assert starts == [0]
        streamer.pc = None

    await _run(streamer, real_sleep, drop)

    assert starts == [0]


@pytest.mark.asyncio
async def test_a_failed_first_connect_is_retried_once_it_succeeds(monkeypatch):
    streamer, starts, real_sleep = _streamer(monkeypatch, first_start_fails=True)

    def drop():
        assert starts == [0, 1]
        streamer.pc = None

    await _run(streamer, real_sleep, drop)

    assert starts == [0, 1]


def _monitored_streamer(monkeypatch, *, failing_starts: int):
    """A streamer whose WebRTC link has just dropped, with the first
    ``failing_starts`` reconnect attempts failing."""
    streamer = _Streamer(
        client=Mock(), twin_uuid="camera-twin", enable_health_check=False
    )
    streamer.pc = SimpleNamespace(connectionState="failed", iceConnectionState="failed")
    streamer._should_reconnect = True
    streamer._is_running = True
    starts: list[int] = []

    async def stop():
        streamer.pc = None

    async def start(twin_uuid=None):
        starts.append(len(starts))
        if len(starts) <= failing_starts:
            raise TimeoutError("Timeout waiting for WebRTC answer")
        streamer.pc = SimpleNamespace(
            connectionState="connected", iceConnectionState="completed"
        )

    monkeypatch.setattr(streamer, "start", start)
    monkeypatch.setattr(streamer, "stop", stop)
    real_sleep = asyncio.sleep

    async def yield_once(_delay):
        await real_sleep(0)

    monkeypatch.setattr(base_video.asyncio, "sleep", yield_once)
    return streamer, starts, real_sleep


async def _monitor(streamer, real_sleep, passes=200, between=None):
    stop = asyncio.Event()
    task = asyncio.create_task(streamer._monitor_connection(stop))
    for i in range(passes):
        if task.done():
            break
        if between is not None and i == passes // 2:
            between()
        await real_sleep(0)
    stop.set()
    await task


@pytest.mark.asyncio
async def test_the_monitor_retries_after_a_failed_reconnect(monkeypatch):
    streamer, starts, real_sleep = _monitored_streamer(monkeypatch, failing_starts=2)

    await _monitor(streamer, real_sleep)

    # Two failed attempts, then the third reconnects; the monitor does not
    # stall on the pc a failed attempt leaves as None.
    assert starts == [0, 1, 2]
    assert streamer.pc is not None
    assert streamer._reconnect_pending is False


@pytest.mark.asyncio
async def test_the_monitor_never_gives_up_on_its_own(monkeypatch):
    streamer, starts, real_sleep = _monitored_streamer(monkeypatch, failing_starts=1000)

    await _monitor(streamer, real_sleep, passes=2000)

    # Past the fast attempts it keeps retrying (slowly), so a stream comes
    # back after an uplink outage of any length.
    assert len(starts) > 10
    assert streamer._should_reconnect is True
    assert streamer._reconnect_pending is True


@pytest.mark.asyncio
async def test_a_stop_command_ends_a_pending_reconnect(monkeypatch):
    streamer, starts, real_sleep = _monitored_streamer(monkeypatch, failing_starts=1000)

    def stop_video():
        # What _handle_stop_command does before it stops the stream.
        streamer._should_reconnect = False
        streamer._reconnect_pending = False

    await _monitor(streamer, real_sleep, passes=40, between=stop_video)
    attempts = len(starts)
    await _monitor(streamer, real_sleep, passes=40)

    assert attempts > 0
    assert len(starts) == attempts
