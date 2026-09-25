"""Tests for the WebRTC reconnect monitor on ``BaseVideoStreamer``.

``start()`` arms the monitor and ``stop()`` ends it. A failed reconnect attempt
is retried. ``run_with_auto_reconnect()`` keeps one monitor and leaves drops to it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

pytest.importorskip("aiortc", reason="aiortc not installed")

from cyberwave.sensor import base_video, hw_encoder  # noqa: E402
from cyberwave.sensor.base_video import BaseVideoStreamer  # noqa: E402


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    # Keep aiortc's encoder global untouched and run every sleep in ms.
    monkeypatch.setattr(
        base_video,
        "apply_h264_hw_patch",
        lambda: hw_encoder.EncoderSelection("libx264", "software"),
    )
    real_sleep = asyncio.sleep

    async def _sleep(seconds: float = 0, *args: Any, **kwargs: Any):
        return await real_sleep(seconds * 0.01, *args, **kwargs)

    monkeypatch.setattr("cyberwave.sensor.base_video.asyncio.sleep", _sleep)


class _FakeMQTT:
    topic_prefix = ""
    client_id = "test-client"

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def publish(self, topic: str, payload: Dict[str, Any], qos: int = 0) -> None:
        self.calls.append((topic, payload))

    def subscribe(self, *args: Any, **kwargs: Any) -> None:
        pass

    def offers(self) -> int:
        return sum(1 for t, _ in self.calls if t.endswith("/webrtc-offer"))


class _FakePC:
    def __init__(self) -> None:
        self.connectionState = "connected"
        self.iceConnectionState = "connected"
        self.localDescription = type("D", (), {"sdp": "v=0\r\n", "type": "offer"})

    async def close(self) -> None:
        self.connectionState = "closed"


class _Track:
    id = "track-1"

    def get_stream_attributes(self) -> Dict[str, Any]:
        return {}

    def close(self) -> None:
        pass


class _Streamer(BaseVideoStreamer):
    fail = False  # set to make every new connection attempt fail
    callers: List[Any]

    def initialize_track(self) -> _Track:
        return _Track()

    async def _setup_webrtc(self) -> None:
        self.callers.append(asyncio.current_task())
        if self.fail:
            raise RuntimeError("network down")
        self.streamer = self.initialize_track()
        self.pc = _FakePC()

    async def _perform_signaling(self) -> None:
        self._send_offer(self.pc.localDescription.sdp)

    async def _wait_and_publish_camera_sync_frame(self, *args: Any, **kwargs: Any):
        return None


def _streamer(auto_reconnect: bool = True) -> _Streamer:
    s = _Streamer(
        client=_FakeMQTT(),
        twin_uuid="twin-1",
        auto_reconnect=auto_reconnect,
        enable_health_check=False,
    )
    s.callers = []
    return s


def _monitors() -> List[asyncio.Task]:
    return [
        t
        for t in asyncio.all_tasks()
        if "_monitor_connection" in repr(t.get_coro()) and not t.done()
    ]


async def _wait_for(predicate, timeout: float = 5.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


async def test_start_arms_the_monitor():
    s = _streamer()
    await s.start()

    assert s._monitor_task is not None and not s._monitor_task.done()
    assert len(_monitors()) == 1

    await s.stop()


async def test_dropped_connection_is_renegotiated():
    s = _streamer()
    await s.start()
    assert s.client.offers() == 1

    s.pc.connectionState = "failed"

    assert await _wait_for(lambda: s.client.offers() == 2)
    assert s.pc.connectionState == "connected"

    await s.stop()


async def test_failed_reconnect_is_retried():
    s = _streamer()
    await s.start()

    s.fail = True
    s.pc.connectionState = "failed"
    assert await _wait_for(lambda: len(s.callers) >= 4)
    assert s.pc is None

    s.fail = False  # network is back
    assert await _wait_for(lambda: s.pc is not None)
    assert s.pc.connectionState == "connected"

    await s.stop()


async def test_stop_ends_the_monitor():
    s = _streamer()
    await s.start()
    await s.stop()

    assert s._monitor_task is None
    assert s.pc is None
    assert _monitors() == []


async def test_start_without_auto_reconnect_has_no_monitor():
    s = _streamer(auto_reconnect=False)
    await s.start()

    assert s._monitor_task is None

    await s.stop()


async def test_run_with_auto_reconnect_keeps_one_monitor():
    s = _streamer()
    stop = asyncio.Event()
    run = asyncio.create_task(s.run_with_auto_reconnect(stop_event=stop))
    assert await _wait_for(lambda: s.pc is not None)

    # A stop_video / start_video pair must not drop the loop's monitor.
    await s._handle_stop_command()
    await s._handle_start_command()
    assert len(_monitors()) == 1

    stop.set()
    await run
    assert _monitors() == []


async def test_manager_leaves_drops_to_the_monitor(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        base_video,
        "time",
        SimpleNamespace(time=base_video.time.time, monotonic=lambda: clock[0]),
    )
    s = _streamer()
    stop = asyncio.Event()
    run = asyncio.create_task(
        s.run_with_auto_reconnect(stop_event=stop, subscribe_to_commands=False)
    )
    assert await _wait_for(lambda: s.pc is not None)

    clock[0] = 100.0  # past the loop's startup retry deadline
    s.fail = True
    s.pc.connectionState = "failed"
    assert await _wait_for(lambda: len(s.callers) >= 4)

    assert all(t is s._monitor_task for t in s.callers[1:])

    stop.set()
    await run


async def test_stop_video_during_reconnect_keeps_the_monitor():
    s = _streamer()
    stop = asyncio.Event()
    run = asyncio.create_task(
        s.run_with_auto_reconnect(stop_event=stop, subscribe_to_commands=False)
    )
    assert await _wait_for(lambda: s.pc is not None)
    monitor = s._monitor_task

    # Catch the monitor right after it closed the dead connection
    closed = asyncio.Event()
    close = s._close_peer_connection

    async def _close():
        await close()
        closed.set()

    s._close_peer_connection = _close
    s.pc.connectionState = "failed"
    await closed.wait()
    await s._handle_stop_command()  # pc is None here
    await s._handle_start_command()
    assert s.pc is not None

    s.pc.connectionState = "failed"
    calls = len(s.callers)
    assert await _wait_for(lambda: len(s.callers) > calls)
    assert not monitor.done()

    stop.set()
    await run


async def test_run_after_start_keeps_one_monitor():
    s = _streamer()
    await s.start()
    stop = asyncio.Event()
    run = asyncio.create_task(
        s.run_with_auto_reconnect(stop_event=stop, subscribe_to_commands=False)
    )
    assert await _wait_for(lambda: s._is_running and s._event_loop is not None)
    await asyncio.sleep(0.05)
    assert len(_monitors()) == 1

    stop.set()
    await run
    assert _monitors() == []


async def test_start_after_retry_limit_reconnects_again():
    s = _streamer()
    await s.start()

    s.fail = True
    s.pc.connectionState = "failed"
    assert await _wait_for(lambda: not s._should_reconnect, timeout=20)

    s.fail = False
    await s.start()
    s.pc.connectionState = "failed"
    calls = len(s.callers)
    assert await _wait_for(lambda: len(s.callers) > calls)

    await s.stop()
