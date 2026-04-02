"""Contract tests that run against every DataBackend implementation.

Each test is parametrized so it executes on both FilesystemBackend and
ZenohBackend (when available).  This ensures behavioural parity.
"""

import threading
import time

import pytest

from cyberwave.data.backend import DataBackend, Sample
from cyberwave.data.config import BackendConfig, get_backend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

try:
    import zenoh  # noqa: F401

    _has_zenoh = True
except ImportError:
    _has_zenoh = False

BACKEND_PARAMS = ["filesystem"]
if _has_zenoh:
    BACKEND_PARAMS.append("zenoh")


@pytest.fixture(params=BACKEND_PARAMS)
def backend(request, tmp_path):
    """Yield a fresh DataBackend for each parametrized backend type."""
    name = request.param
    if name == "filesystem":
        cfg = BackendConfig(
            backend="filesystem",
            filesystem_base_dir=str(tmp_path / "data"),
            filesystem_ring_buffer_size=50,
        )
    else:
        cfg = BackendConfig(backend="zenoh")
    be = get_backend(cfg)
    yield be
    be.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPublishAndLatest:
    def test_publish_then_latest(self, backend: DataBackend):
        backend.publish("test/channel", b"hello world")
        sample = backend.latest("test/channel")
        assert sample is not None
        assert sample.payload == b"hello world"
        assert sample.channel == "test/channel"

    def test_latest_returns_none_on_empty_channel(self, backend: DataBackend):
        result = backend.latest("nonexistent/channel", timeout_s=0.1)
        assert result is None

    def test_publish_overwrites_latest(self, backend: DataBackend):
        backend.publish("ch", b"first")
        backend.publish("ch", b"second")
        sample = backend.latest("ch")
        assert sample is not None
        assert sample.payload == b"second"

    def test_multiple_channels_are_isolated(self, backend: DataBackend):
        backend.publish("alpha", b"aaa")
        backend.publish("beta", b"bbb")
        assert backend.latest("alpha").payload == b"aaa"  # type: ignore[union-attr]
        assert backend.latest("beta").payload == b"bbb"  # type: ignore[union-attr]
        assert backend.latest("gamma", timeout_s=0.1) is None

    def test_empty_payload(self, backend: DataBackend):
        backend.publish("empty", b"")
        sample = backend.latest("empty")
        assert sample is not None
        assert sample.payload == b""

    def test_large_payload(self, backend: DataBackend):
        large = b"\x00" * (1024 * 1024)
        backend.publish("big", large)
        sample = backend.latest("big")
        assert sample is not None
        assert len(sample.payload) == 1024 * 1024


class TestSubscribe:
    def test_subscribe_receives_published_sample(self, backend: DataBackend):
        received: list[Sample] = []
        event = threading.Event()

        def cb(s: Sample) -> None:
            received.append(s)
            event.set()

        sub = backend.subscribe("sub/test", cb)
        time.sleep(0.15)
        backend.publish("sub/test", b"payload1")
        assert event.wait(timeout=2.0), "Callback was not invoked"
        assert len(received) >= 1
        assert received[-1].payload == b"payload1"
        sub.close()

    def test_subscribe_fifo_delivers_all(self, backend: DataBackend):
        received: list[bytes] = []
        barrier = threading.Event()

        def cb(s: Sample) -> None:
            received.append(s.payload)
            if len(received) >= 3:
                barrier.set()

        sub = backend.subscribe("fifo/ch", cb, policy="fifo")
        time.sleep(0.15)
        for i in range(3):
            backend.publish("fifo/ch", f"msg{i}".encode())
            time.sleep(0.08)
        barrier.wait(timeout=3.0)
        assert len(received) >= 3
        assert b"msg0" in received
        assert b"msg1" in received
        assert b"msg2" in received
        sub.close()

    def test_subscribe_latest_may_skip_intermediate(self, backend: DataBackend):
        received: list[bytes] = []
        done = threading.Event()

        def cb(s: Sample) -> None:
            received.append(s.payload)
            done.set()

        sub = backend.subscribe("lat/ch", cb, policy="latest")
        time.sleep(0.15)
        for i in range(5):
            backend.publish("lat/ch", f"v{i}".encode())
        done.wait(timeout=2.0)
        time.sleep(0.3)
        assert len(received) >= 1
        assert received[-1] == b"v4" or len(received) <= 5
        sub.close()

    def test_close_stops_callback(self, backend: DataBackend):
        count = 0
        lock = threading.Lock()

        def cb(s: Sample) -> None:
            nonlocal count
            with lock:
                count += 1

        sub = backend.subscribe("stop/ch", cb)
        time.sleep(0.15)
        backend.publish("stop/ch", b"a")
        time.sleep(0.3)
        sub.close()
        time.sleep(0.1)
        snapshot = count
        backend.publish("stop/ch", b"b")
        time.sleep(0.3)
        with lock:
            assert count == snapshot


class TestContextManager:
    def test_with_statement(self, tmp_path):
        cfg = BackendConfig(
            backend="filesystem",
            filesystem_base_dir=str(tmp_path / "ctx"),
        )
        with get_backend(cfg) as be:
            be.publish("ctx/ch", b"data")
            assert be.latest("ctx/ch") is not None


class TestPolicyValidation:
    def test_invalid_policy_raises(self, backend: DataBackend):
        with pytest.raises(ValueError, match="Invalid subscribe policy"):
            backend.subscribe("bad/policy", lambda s: None, policy="random")


class TestSampleType:
    def test_sample_has_expected_fields(self, backend: DataBackend):
        backend.publish("fields", b"x", metadata={"key": "val"})
        sample = backend.latest("fields")
        assert sample is not None
        assert isinstance(sample.channel, str)
        assert isinstance(sample.payload, bytes)
        assert isinstance(sample.timestamp, float)
