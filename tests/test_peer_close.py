"""A peer-connection close must never hold a stop past its bound."""

import asyncio
import logging
import time

import pytest

from cyberwave.sensor import peer_close
from cyberwave.sensor.peer_close import close_peer_connection


class _Pc:
    def __init__(self, close):
        self.close = close


@pytest.mark.asyncio
async def test_hanging_close_is_abandoned_after_the_timeout():
    async def hang():
        await asyncio.Event().wait()

    started = time.monotonic()
    closed = await close_peer_connection(_Pc(hang), timeout=0.1, settle=5.0)
    assert closed is False
    # The settle is skipped too: nothing settled, waiting would not help.
    assert time.monotonic() - started < 1.0


@pytest.mark.asyncio
async def test_failing_close_is_reported_not_raised():
    async def fail():
        raise RuntimeError("transport already gone")

    assert await close_peer_connection(_Pc(fail), timeout=1.0, settle=0) is False


@pytest.mark.asyncio
async def test_clean_close_settles_then_reports_success():
    calls = []

    async def close():
        calls.append("close")

    started = time.monotonic()
    assert await close_peer_connection(_Pc(close), timeout=1.0, settle=0.05) is True
    assert calls == ["close"]
    assert time.monotonic() - started >= 0.05


@pytest.mark.asyncio
async def test_failing_close_logs_at_error_by_default(caplog):
    async def fail():
        raise RuntimeError("transport already gone")

    with caplog.at_level(logging.DEBUG, logger="cyberwave.sensor.peer_close"):
        await close_peer_connection(_Pc(fail), timeout=1.0, settle=0)

    assert [r.levelno for r in caplog.records] == [logging.ERROR]


@pytest.mark.asyncio
async def test_failing_close_logs_at_the_requested_level(caplog):
    async def fail():
        raise RuntimeError("transport already gone")

    with caplog.at_level(logging.DEBUG, logger="cyberwave.sensor.peer_close"):
        await close_peer_connection(
            _Pc(fail), timeout=1.0, settle=0, error_log_level=logging.DEBUG
        )

    assert [r.levelno for r in caplog.records] == [logging.DEBUG]
    assert caplog.records[0].exc_info is not None


def test_close_budget_outlasts_the_close_it_wraps(monkeypatch):
    monkeypatch.setattr(peer_close, "PEER_CONNECTION_CLOSE_TIMEOUT_S", 0.2)
    monkeypatch.setattr(peer_close, "PEER_CONNECTION_SETTLE_S", 0.3)

    # Read at call time, so a patched cap moves the outer deadline too.
    assert peer_close.close_budget_s() == pytest.approx(0.2 + 0.3 + 1.0)
    assert peer_close.close_budget_s(settle=0) == pytest.approx(0.2 + 1.0)
    assert peer_close.close_budget_s(extra=1.5) == pytest.approx(0.2 + 0.3 + 1.5 + 1.0)
