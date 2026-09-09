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
