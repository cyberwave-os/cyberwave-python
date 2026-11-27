"""Tolerant ICE consent check: the patch, its thresholds and its reports."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

aioice = pytest.importorskip("aioice", reason="aioice not installed (install with extras: camera)")

from aioice import ice, stun  # noqa: E402

from cyberwave.sensor import ice_consent  # noqa: E402
from cyberwave.sensor.ice_consent import (  # noqa: E402
    CONSENT_FAILURES,
    CONSENT_RETRANSMISSIONS,
    ConsentWatch,
    apply_ice_consent_patch,
)


@pytest.fixture(autouse=True)
def _patched() -> None:
    assert apply_ice_consent_patch() is True


class _Pair:
    """A nominated pair whose STUN requests fail ``misses`` times, then succeed."""

    def __init__(self, misses: int) -> None:
        self.misses = misses
        self.requests = 0
        self.retransmissions: list[Any] = []
        self.remote_addr = ("192.0.2.1", 3478)
        self.protocol = self

    async def request(self, request: Any, addr: Any, integrity_key: Any, retransmissions: Any) -> Any:
        self.requests += 1
        self.retransmissions.append(retransmissions)
        if self.requests <= self.misses:
            raise stun.TransactionTimeout()
        return None, addr


class _Connection:
    """The parts of ``aioice.Connection`` the consent loop touches."""

    def __init__(self, pair: _Pair) -> None:
        self._nominated = {1: pair}
        self.remote_password = "remote-password-0123456789"
        self._query_consent_task: Any = "running"
        self.closed = 0

    def build_request(self, pair: Any, nominate: bool) -> Any:
        return object()

    async def close(self) -> None:
        self.closed += 1

    def _Connection__log_info(self, msg: str, *args: Any) -> None:
        """aioice's own loop logs through this name-mangled method."""


async def _run_consent(loop_fn: Any, connection: _Connection, pair: _Pair, max_requests: int) -> None:
    task = asyncio.ensure_future(loop_fn(connection))
    while not task.done() and pair.requests < max_requests:
        await asyncio.sleep(0)
    if not task.done():
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.fixture
def no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ice, "CONSENT_INTERVAL", 0)


def test_patch_is_applied_once_and_is_idempotent() -> None:
    patched = ice.Connection.query_consent
    assert getattr(patched, "_cyberwave_tolerant_consent", False)
    assert ice.CONSENT_FAILURES == CONSENT_FAILURES == 12

    assert apply_ice_consent_patch() is True
    assert ice.Connection.query_consent is patched
    # The original is kept, and is not wrapped twice.
    assert not getattr(patched.__wrapped__, "_cyberwave_tolerant_consent", False)


async def test_six_misses_no_longer_close_the_connection(no_wait: None) -> None:
    pair = _Pair(misses=6)
    connection = _Connection(pair)
    await _run_consent(ice.Connection.query_consent, connection, pair, max_requests=20)
    assert connection.closed == 0
    assert set(pair.retransmissions) == {CONSENT_RETRANSMISSIONS}


async def test_aioice_default_closes_after_six_misses(
    no_wait: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Contrast: aioice's own loop with its own threshold.
    monkeypatch.setattr(ice, "CONSENT_FAILURES", 6)
    pair = _Pair(misses=6)
    connection = _Connection(pair)
    await _run_consent(ice.Connection.query_consent.__wrapped__, connection, pair, max_requests=20)
    assert connection.closed == 1
    assert pair.requests == 6
    assert set(pair.retransmissions) == {0}


async def test_new_threshold_closes_and_reports(
    no_wait: None, caplog: pytest.LogCaptureFixture
) -> None:
    pair = _Pair(misses=10**6)
    connection = _Connection(pair)
    watch = ConsentWatch(label="twin T camera C", tracked=True)
    connection._cyberwave_consent_handlers = [watch]  # type: ignore[attr-defined]

    with caplog.at_level(logging.WARNING, logger="cyberwave.sensor.ice_consent"):
        await _run_consent(ice.Connection.query_consent, connection, pair, max_requests=100)

    assert pair.requests == CONSENT_FAILURES
    assert connection.closed == 1
    assert connection._query_consent_task is None
    assert watch.expired_at is not None
    assert "ICE consent expired for twin T camera C" in caplog.text
    assert watch.describe_close().startswith("after ICE consent expired")


async def test_a_success_resets_the_count(no_wait: None) -> None:
    class _Flaky(_Pair):
        async def request(self, *args: Any, **kwargs: Any) -> Any:
            self.requests += 1
            # 11 misses, 1 answer, 11 misses: never 12 in a row.
            if self.requests % 12 == 0:
                return None, self.remote_addr
            raise stun.TransactionTimeout()

    pair = _Flaky(misses=0)
    connection = _Connection(pair)
    await _run_consent(ice.Connection.query_consent, connection, pair, max_requests=23)
    assert connection.closed == 0


def test_describe_close_without_events() -> None:
    assert ConsentWatch(label="x").describe_close() == "ICE consent expiry not tracked"
    assert (
        ConsentWatch(label="x", tracked=True).describe_close()
        == "no ICE consent expiry or socket loss recorded"
    )


def test_unsupported_aioice_is_a_noop_with_one_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    patched = ice.Connection.query_consent
    monkeypatch.setitem(ice_consent._state, "result", None)
    monkeypatch.setattr(ice.Connection, "query_consent", patched.__wrapped__)
    monkeypatch.setattr(aioice, "__version__", "9.9.0")

    with caplog.at_level(logging.WARNING, logger="cyberwave.sensor.ice_consent"):
        assert apply_ice_consent_patch() is False
        assert apply_ice_consent_patch() is False
    assert caplog.text.count("ICE consent patch not applied") == 1
    assert ice.Connection.query_consent is patched.__wrapped__


def test_changed_query_consent_is_a_noop(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def query_consent(self: Any) -> None:  # a rewritten aioice loop
        await asyncio.sleep(1)

    monkeypatch.setitem(ice_consent._state, "result", None)
    monkeypatch.setattr(ice.Connection, "query_consent", query_consent)
    with caplog.at_level(logging.WARNING, logger="cyberwave.sensor.ice_consent"):
        assert apply_ice_consent_patch() is False
    assert "query_consent changed" in caplog.text
    assert ice.Connection.query_consent is query_consent


async def test_socket_lost_while_connected_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    class _Receiver:
        def __init__(self) -> None:
            self._nominated: dict[int, Any] = {}
            self.lost: list[Any] = []

        # aioice < 0.11 and >= 0.11 respectively.
        def data_received(self, data: Any, component: Any) -> None:
            self.lost.append(data)

        def protocol_lost(self, protocol: Any) -> None:
            self.lost.append(protocol)

    receiver = _Receiver()
    media = ice.StunProtocol(receiver)  # type: ignore[arg-type]
    relay = ice.StunProtocol(receiver)  # type: ignore[arg-type]

    class _P:
        protocol = media

    receiver._nominated = {1: _P()}
    watch = ConsentWatch(label="twin T camera C", tracked=True)
    receiver._cyberwave_consent_handlers = [watch]  # type: ignore[attr-defined]

    with caplog.at_level(logging.WARNING, logger="cyberwave.sensor.ice_consent"):
        relay.connection_lost(ConnectionResetError("reset by peer"))

    assert receiver.lost, "aioice's own handling still runs"
    assert watch.socket_lost_at is not None
    assert "not the media pair" in watch.socket_lost_detail
    assert "ICE socket lost for twin T camera C while connected" in caplog.text
    assert "socket" in watch.describe_close()

    # A normal close clears the nominated pairs first: nothing to report.
    receiver._nominated = {}
    watch.socket_lost_at = None
    media.connection_lost(None)
    assert watch.socket_lost_at is None


async def test_watch_attaches_to_a_real_peer_connection() -> None:
    aiortc = pytest.importorskip("aiortc")
    from aiortc.mediastreams import VideoStreamTrack

    pc = aiortc.RTCPeerConnection()
    try:
        pc.addTrack(VideoStreamTrack())
        watch = ice_consent.watch_consent_expiry(pc, "twin T camera C")
        assert watch.tracked
        connections = ice_consent._ice_connections(pc)
        assert connections
        assert watch in connections[0]._cyberwave_consent_handlers
    finally:
        await pc.close()
