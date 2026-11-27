"""Event-loop lag probe: warns on a blocked loop, at most once per window."""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from cyberwave.sensor.loop_lag import EventLoopLagProbe

LOGGER = "cyberwave.sensor.loop_lag"


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_no_warning_without_lag(caplog: pytest.LogCaptureFixture) -> None:
    probe = EventLoopLagProbe("twin T camera C", clock=_Clock())
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for lag in (0.0, 0.01, 0.1, 0.25):
            probe.record(lag)
    assert caplog.records == []
    assert probe.max_lag == 0.25


def test_warns_once_per_window_with_the_max_lag(caplog: pytest.LogCaptureFixture) -> None:
    clock = _Clock()
    probe = EventLoopLagProbe("twin T camera C", clock=clock)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        probe.record(0.3)  # first late wake-up: reported at once
        for lag in (0.9, 0.4, 1.2, 0.5):
            clock.now += 1
            probe.record(lag)  # inside the 30 s window: silent
        clock.now += 30
        probe.record(0.26)  # next window: reports the max seen since the last report

    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 2
    assert "twin T camera C" in messages[0]
    assert "woke 300 ms late" in messages[0]
    assert "max 1200 ms over 5 late wake-ups" in messages[1]
    assert probe.max_lag == 1.2


async def test_probe_measures_a_blocked_loop(caplog: pytest.LogCaptureFixture) -> None:
    probe = EventLoopLagProbe("twin T camera C", interval=0.05, threshold=0.1)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        probe.start()
        probe.start()  # idempotent
        await asyncio.sleep(0.07)
        time.sleep(0.3)  # block the loop, as heavy per-frame work would
        await asyncio.sleep(0.1)
        await probe.stop()
    assert not probe.running
    assert probe.max_lag >= 0.15
    assert "Event loop lag for twin T camera C" in caplog.text


async def test_probe_quiet_on_an_idle_loop(caplog: pytest.LogCaptureFixture) -> None:
    probe = EventLoopLagProbe("twin T camera C", interval=0.02, threshold=0.25)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        probe.start()
        await asyncio.sleep(0.2)
        await probe.stop()
        await probe.stop()  # idempotent
    assert caplog.records == []


async def test_probe_never_spins_when_asyncio_sleep_is_patched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests patch asyncio.sleep with mocks that return without yielding.

    A probe awaiting asyncio.sleep would then spin on the loop forever; the
    timer-based probe must not depend on it.
    """
    from unittest.mock import AsyncMock

    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    probe = EventLoopLagProbe("twin T camera C", interval=0.01)
    probe.start()
    # Other work still runs: this only completes if the probe yields the loop.
    await asyncio.wait_for(real_sleep(0.05), timeout=2)
    await probe.stop()
    assert not probe.running
