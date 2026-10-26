"""Canonical camera runner must stop even while signaling never answers."""

import asyncio
import threading
from types import SimpleNamespace

from cyberwave.sensor.manager import run_streamer_in_background


def test_stop_interrupts_pending_offer_and_closes_partial_stream():
    started, cancelled, closed = threading.Event(), threading.Event(), threading.Event()

    class Streamer:
        client = SimpleNamespace(connected=True)

        async def run_with_auto_reconnect(self, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()  # Unreachable signaling server.
            finally:
                cancelled.set()

        async def stop(self):
            closed.set()

    stop = threading.Event()
    thread = run_streamer_in_background(Streamer(), stop)
    assert started.wait(1)
    stop.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert cancelled.is_set() and closed.is_set()


def test_normal_stop_preserves_cooperative_cleanup():
    started, cleaned = threading.Event(), threading.Event()

    class Streamer:
        client = SimpleNamespace(connected=True)

        async def run_with_auto_reconnect(self, *, stop_event, **kwargs):
            started.set()
            await stop_event.wait()
            cleaned.set()

        async def stop(self):
            raise AssertionError("Cooperative shutdown already owns cleanup")

    stop = threading.Event()
    thread = run_streamer_in_background(Streamer(), stop)
    assert started.wait(1)
    stop.set()
    thread.join(timeout=2)
    assert not thread.is_alive() and cleaned.is_set()


def test_wedged_forced_stop_is_bounded(monkeypatch):
    """A streamer whose stop() never returns must not pin the thread forever."""
    from cyberwave.sensor import manager

    monkeypatch.setattr(manager, "STREAMER_STOP_TIMEOUT_S", 0.2)
    started = threading.Event()

    class Streamer:
        client = SimpleNamespace(connected=True)

        async def run_with_auto_reconnect(self, **kwargs):
            started.set()
            await asyncio.Event().wait()

        async def stop(self):
            await asyncio.Event().wait()  # pc.close() stuck on a dead relay.

    stop = threading.Event()
    thread = run_streamer_in_background(Streamer(), stop)
    assert started.wait(1)
    stop.set()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_stop_during_mqtt_wait_exits_without_streaming():
    class Streamer:
        client = SimpleNamespace(connected=False, connect=lambda: None)

        async def run_with_auto_reconnect(self, **kwargs):
            raise AssertionError("Must not start after stop")

    stop = threading.Event()
    stop.set()
    thread = run_streamer_in_background(Streamer(), stop)
    thread.join(timeout=2)
    assert not thread.is_alive()
