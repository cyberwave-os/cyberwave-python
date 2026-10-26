"""Subscriber-presence API: default behaviour, DataBus keys, and real Zenoh.

The real-transport tests open two genuine Zenoh sessions over explicit
loopback TCP (multicast scouting off on the subscriber side), and are skipped
when ``eclipse-zenoh`` is not installed.  Every session, subscriber and
listener is closed explicitly; waits are short polls with a hard deadline so a
failure can never hang CI.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from typing import Any, Callable

import pytest

from cyberwave.data import DataBackend, DataBus, Sample, Subscription
from cyberwave.data.filesystem_backend import FilesystemBackend
from cyberwave.data.keys import build_key

try:
    import zenoh

    _has_zenoh = True
except ImportError:  # pragma: no cover
    _has_zenoh = False

requires_zenoh = pytest.mark.skipif(
    not _has_zenoh, reason="eclipse-zenoh not installed"
)


TWIN_1 = "11111111-1111-4111-8111-111111111111"
TWIN_2 = "22222222-2222-4222-8222-222222222222"


# --------------------------------------------------------------------------
# Default (non-Zenoh) behaviour and DataBus key plumbing
# --------------------------------------------------------------------------


class _MinimalBackend(DataBackend):
    """A backend that predates the presence API (implements only the ABC)."""

    def publish(self, channel: str, payload: bytes, *, metadata: Any = None) -> None:
        pass

    def subscribe(self, channel, callback, *, policy="latest") -> Subscription:
        return Subscription()

    def latest(self, channel: str, *, timeout_s: float = 1.0) -> Sample | None:
        return None

    def close(self) -> None:
        pass


def test_default_backend_reports_unknown_and_raises_for_callbacks():
    backend = _MinimalBackend()
    assert backend.has_subscribers("any") is None
    with pytest.raises(NotImplementedError):
        backend.on_subscribers_changed("any", lambda _present: None)


def test_filesystem_backend_reports_unknown(tmp_path):
    backend = FilesystemBackend(base_dir=tmp_path)
    try:
        assert backend.has_subscribers("frames/default") is None
        with pytest.raises(NotImplementedError):
            backend.on_subscribers_changed("frames/default", lambda _p: None)
    finally:
        backend.close()


class _RecordingBackend(_MinimalBackend):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bool]] = []

    def has_subscribers(self, channel, *, local_only=False):
        self.calls.append(("has", channel, local_only))
        return True

    def on_subscribers_changed(self, channel, callback, *, local_only=False):
        self.calls.append(("on", channel, local_only))
        return Subscription()


def test_databus_resolves_the_same_key_as_publish_and_subscribe():
    backend = _RecordingBackend()
    bus = DataBus(backend, twin_uuid=TWIN_1)
    expected = build_key(TWIN_1, "frames/s1", None, prefix="cw")
    assert bus.has_subscribers("frames/s1") is True
    bus.on_subscribers_changed("frames/s1", lambda _p: None, local_only=True)
    assert backend.calls == [("has", expected, False), ("on", expected, True)]

    other = build_key(TWIN_2, "frames/s1", None, prefix="cw")
    bus.has_subscribers("frames/s1", twin_uuid=TWIN_2)
    assert backend.calls[-1] == ("has", other, False)


def test_databus_over_a_minimal_backend_is_unknown():
    bus = DataBus(_MinimalBackend(), twin_uuid=TWIN_1)
    assert bus.has_subscribers("frames/s1") is None


# --------------------------------------------------------------------------
# Real two-session Zenoh
# --------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for(predicate: Callable[[], bool], timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


@pytest.fixture
def two_sessions():
    """``(backend, peer_session, channel)``: a ZenohBackend listening on a free
    loopback port and a raw second session (multicast off) connected to it."""
    from cyberwave.data.zenoh_backend import ZenohBackend

    port = _free_port()
    backend = ZenohBackend(listen=[f"tcp/127.0.0.1:{port}"], shared_memory=False)
    cfg = zenoh.Config()
    cfg.insert_json5("scouting/multicast/enabled", "false")
    cfg.insert_json5("connect/endpoints", json.dumps([f"tcp/127.0.0.1:{port}"]))
    cfg.insert_json5("transport/shared_memory/enabled", "false")
    peer = zenoh.open(cfg)
    time.sleep(0.3)  # let the link come up
    channel = f"test/presence/{uuid.uuid4().hex}"
    try:
        yield backend, peer, channel
    finally:
        for closer in (backend.close, peer.close):
            try:
                closer()
            except Exception:
                pass


@requires_zenoh
def test_has_subscribers_flips_with_subscribe_and_unsubscribe(two_sessions):
    backend, peer, channel = two_sessions
    assert backend.has_subscribers(channel) is False

    sub = peer.declare_subscriber(channel, lambda _s: None)
    try:
        assert _wait_for(lambda: backend.has_subscribers(channel) is True)
    finally:
        sub.undeclare()
    assert _wait_for(lambda: backend.has_subscribers(channel) is False)


@requires_zenoh
def test_wildcard_subscriber_matches(two_sessions):
    backend, peer, channel = two_sessions
    prefix = channel.rsplit("/", 1)[0]
    sub = peer.declare_subscriber(f"{prefix}/**", lambda _s: None)
    try:
        assert _wait_for(lambda: backend.has_subscribers(channel) is True)
    finally:
        sub.undeclare()


@requires_zenoh
def test_a_queryable_alone_does_not_count(two_sessions):
    backend, peer, channel = two_sessions
    qable = peer.declare_queryable(channel, lambda q: None)
    try:
        time.sleep(0.5)  # generous: a queryable must NOT flip it
        assert backend.has_subscribers(channel) is False
    finally:
        qable.undeclare()


@requires_zenoh
def test_publishing_data_still_works_and_does_not_count_as_subscriber(two_sessions):
    backend, peer, channel = two_sessions
    got: list[bytes] = []
    backend.has_subscribers(channel)  # declares the presence publisher
    backend.publish(channel, b"x")  # the data path is session.put, unchanged
    assert backend.has_subscribers(channel) is False
    sub = peer.declare_subscriber(channel, lambda s: got.append(bytes(s.payload)))
    try:
        assert _wait_for(lambda: backend.has_subscribers(channel) is True)
        backend.publish(channel, b"y")
        assert _wait_for(lambda: got == [b"y"])
    finally:
        sub.undeclare()


@requires_zenoh
def test_on_subscribers_changed_fires_on_both_edges(two_sessions):
    backend, peer, channel = two_sessions
    events: list[bool] = []
    flipped = threading.Event()

    def _cb(present: bool) -> None:
        events.append(present)
        flipped.set()

    handle = backend.on_subscribers_changed(channel, _cb)
    try:
        assert backend.has_subscribers(channel) is False  # initial state, no call
        assert events == []
        sub = peer.declare_subscriber(channel, lambda _s: None)
        assert _wait_for(lambda: events == [True])
        sub.undeclare()
        assert _wait_for(lambda: events == [True, False])
    finally:
        handle.close()
        handle.close()  # idempotent
    # After close the callback is silent.
    sub = peer.declare_subscriber(channel, lambda _s: None)
    time.sleep(0.4)
    sub.undeclare()
    assert events == [True, False]


@requires_zenoh
def test_a_raising_callback_does_not_break_the_listener(two_sessions):
    backend, peer, channel = two_sessions
    calls: list[bool] = []

    def _cb(present: bool) -> None:
        calls.append(present)
        raise RuntimeError("boom")

    handle = backend.on_subscribers_changed(channel, _cb)
    try:
        sub = peer.declare_subscriber(channel, lambda _s: None)
        assert _wait_for(lambda: calls == [True])
        sub.undeclare()
        assert _wait_for(lambda: calls == [True, False])
    finally:
        handle.close()


@requires_zenoh
def test_local_only_ignores_other_sessions(two_sessions):
    """Zenoh locality is per *session*: a subscriber in another process (here a
    second session) is remote even on the same host, so ``local_only`` does not
    see it.  It sees a subscriber declared on the backend's own session."""
    backend, peer, channel = two_sessions
    sub = peer.declare_subscriber(channel, lambda _s: None)
    try:
        assert _wait_for(lambda: backend.has_subscribers(channel) is True)
        time.sleep(0.2)
        assert backend.has_subscribers(channel, local_only=True) is False
    finally:
        sub.undeclare()

    own = backend.subscribe(channel, lambda _s: None)
    try:
        assert _wait_for(
            lambda: backend.has_subscribers(channel, local_only=True) is True
        )
    finally:
        own.close()


@requires_zenoh
def test_close_releases_presence_resources_and_degrades_to_unknown(two_sessions):
    backend, _peer, channel = two_sessions
    handle = backend.on_subscribers_changed(channel, lambda _p: None)
    assert backend.has_subscribers(channel) is False
    backend.close()
    assert backend.has_subscribers(channel) is None
    handle.close()  # safe after backend close
