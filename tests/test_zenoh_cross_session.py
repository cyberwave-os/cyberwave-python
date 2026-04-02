"""Cross-session Zenoh tests: latest-across-sessions, fan-out, policy semantics.

All tests are skipped when ``eclipse-zenoh`` is not installed.  They use an
in-process broker (same pattern as ``examples/zenoh_triad.py``) so they work
reliably without relying on multicast scouting, e.g. in WSL2 or containers.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any

import pytest

try:
    import zenoh  # noqa: F401

    _has_zenoh = True
except ImportError:
    _has_zenoh = False

pytestmark = pytest.mark.skipif(not _has_zenoh, reason="eclipse-zenoh not installed")


# ---------------------------------------------------------------------------
# Broker / backend helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_broker(port: int) -> Any:
    import zenoh

    cfg = zenoh.Config()
    cfg.insert_json5("listen/endpoints", json.dumps([f"tcp/127.0.0.1:{port}"]))
    cfg.insert_json5("transport/shared_memory/enabled", "false")
    return zenoh.open(cfg)


def _make_backend(connect: list[str]) -> Any:
    from cyberwave.data.zenoh_backend import ZenohBackend

    return ZenohBackend(key_prefix="", connect=connect, shared_memory=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def broker():
    """Start an in-process Zenoh broker; yield the connect endpoint list."""
    port = _find_free_port()
    session = _start_broker(port)
    time.sleep(0.15)  # give the broker time to start accepting connections
    yield [f"tcp/127.0.0.1:{port}"]
    try:
        session.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# TestLatestAcrossSession
# ---------------------------------------------------------------------------


class TestLatestAcrossSession:
    """Session B (late-joining) can query the latest value from Session A.

    ``ZenohBackend.latest()`` works by issuing a Zenoh ``get()`` that is
    answered by the queryable declared inside the publishing session.  The
    publisher session must remain alive for the queryable to be available;
    once it closes the queryable is undeclared.  These tests cover the common
    real-world pattern: a long-running publisher (A) and a late-joining reader
    (B) that opens its own independent session after the first publish.
    """

    def test_late_joining_session_gets_latest(self, broker: list[str]) -> None:
        """Session B opened after A published can still retrieve the value."""
        be_a = _make_backend(broker)
        be_a.publish("cross/ch", b"hello-cross")
        time.sleep(0.3)

        # Session B opens after A has already published.
        be_b = _make_backend(broker)
        try:
            sample = be_b.latest("cross/ch", timeout_s=2.0)
            assert sample is not None, "late-joining session got None from latest()"
            assert sample.payload == b"hello-cross"
        finally:
            be_b.close()
            be_a.close()

    def test_latest_returns_most_recent_value(self, broker: list[str]) -> None:
        """Session B sees the most recently published value, not an older one."""
        be_a = _make_backend(broker)
        for v in (b"v1", b"v2", b"v3"):
            be_a.publish("latest/seq", v)
            time.sleep(0.05)

        # Session B opened after all three publishes.
        be_b = _make_backend(broker)
        try:
            sample = be_b.latest("latest/seq", timeout_s=2.0)
            assert sample is not None
            assert sample.payload == b"v3"
        finally:
            be_b.close()
            be_a.close()


# ---------------------------------------------------------------------------
# TestFanOut
# ---------------------------------------------------------------------------


class TestFanOut:
    """One publisher, two independent subscriber sessions — both receive all messages."""

    def test_two_subscribers_receive_all(self, broker: list[str]) -> None:
        n = 5
        received_b: list[bytes] = []
        received_c: list[bytes] = []
        done_b = threading.Event()
        done_c = threading.Event()

        be_b = _make_backend(broker)
        be_c = _make_backend(broker)

        def cb_b(s: Any) -> None:
            received_b.append(s.payload)
            if len(received_b) >= n:
                done_b.set()

        def cb_c(s: Any) -> None:
            received_c.append(s.payload)
            if len(received_c) >= n:
                done_c.set()

        sub_b = be_b.subscribe("fanout/ch", cb_b, policy="fifo")
        sub_c = be_c.subscribe("fanout/ch", cb_c, policy="fifo")

        # Give subscriptions time to be declared before publishing.
        time.sleep(0.3)

        be_a = _make_backend(broker)
        for i in range(n):
            be_a.publish("fanout/ch", f"msg{i}".encode())
            time.sleep(0.05)
        be_a.close()

        assert done_b.wait(timeout=5.0), f"B only received {len(received_b)}/{n}"
        assert done_c.wait(timeout=5.0), f"C only received {len(received_c)}/{n}"

        sub_b.close()
        sub_c.close()
        be_b.close()
        be_c.close()

        assert len(received_b) == n
        assert len(received_c) == n

    def test_closing_one_subscriber_does_not_affect_other(
        self, broker: list[str]
    ) -> None:
        received_c: list[bytes] = []
        done_c = threading.Event()

        be_b = _make_backend(broker)
        be_c = _make_backend(broker)

        sub_b = be_b.subscribe("fanout2/ch", lambda s: None, policy="fifo")
        sub_c = be_c.subscribe(
            "fanout2/ch",
            lambda s: (received_c.append(s.payload), done_c.set() if len(received_c) >= 3 else None),  # type: ignore[func-returns-value]
            policy="fifo",
        )
        time.sleep(0.3)

        # Close B before A publishes.
        sub_b.close()
        be_b.close()

        be_a = _make_backend(broker)
        for i in range(3):
            be_a.publish("fanout2/ch", f"x{i}".encode())
            time.sleep(0.05)
        be_a.close()

        assert done_c.wait(timeout=5.0), f"C only received {len(received_c)}/3"
        sub_c.close()
        be_c.close()


# ---------------------------------------------------------------------------
# TestPolicySemantics
# ---------------------------------------------------------------------------


class TestPolicySemantics:
    """``"fifo"`` delivers all messages; ``"latest"`` may skip intermediates."""

    def test_fifo_delivers_all(self, broker: list[str]) -> None:
        n = 10
        received: list[bytes] = []
        done = threading.Event()

        be = _make_backend(broker)

        def cb(s: Any) -> None:
            received.append(s.payload)
            if len(received) >= n:
                done.set()

        sub = be.subscribe("policy/fifo", cb, policy="fifo")
        time.sleep(0.2)

        for i in range(n):
            be.publish("policy/fifo", f"v{i}".encode())
            time.sleep(0.02)

        done.wait(timeout=5.0)
        sub.close()
        be.close()

        assert len(received) == n, f"fifo: expected {n} but got {len(received)}"
        for i in range(n):
            assert f"v{i}".encode() in received

    def test_latest_may_skip_intermediates(self, broker: list[str]) -> None:
        """Under flood conditions the ``"latest"`` policy may drop intermediate
        messages.  We only assert that: (a) at least one message arrived,
        (b) no more messages than published arrived, and (c) the last message
        is guaranteed to have been received."""
        n = 20
        received: list[bytes] = []
        got_any = threading.Event()

        be = _make_backend(broker)

        def cb(s: Any) -> None:
            received.append(s.payload)
            got_any.set()

        sub = be.subscribe("policy/latest", cb, policy="latest")
        time.sleep(0.2)

        for i in range(n):
            be.publish("policy/latest", f"v{i}".encode())

        # Give the subscriber time to drain.
        time.sleep(0.5)
        sub.close()
        be.close()

        assert got_any.is_set(), "latest subscriber received nothing"
        assert len(received) <= n
        assert f"v{n - 1}".encode() in received, "last message must be received"
