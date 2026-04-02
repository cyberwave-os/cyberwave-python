"""Tests for cyberwave.data.api — DataBus public API facade.

Uses the FilesystemBackend for real end-to-end roundtrips (no mocks for
the transport layer), keeping tests simple and deterministic.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest

from cyberwave.data.api import DataBus
from cyberwave.data.backend import Sample
from cyberwave.data.filesystem_backend import FilesystemBackend

TWIN_UUID = "550e8400-e29b-41d4-a716-446655440000"


@pytest.fixture()
def backend(tmp_path):
    """Fresh FilesystemBackend rooted in a temporary directory."""
    be = FilesystemBackend(base_dir=tmp_path, poll_interval_s=0.02)
    yield be
    be.close()


@pytest.fixture()
def bus(backend):
    """DataBus wired to the temp backend."""
    return DataBus(backend, TWIN_UUID, key_prefix="cw")


# ── publish + latest roundtrips ──────────────────────────────────────


class TestPublishLatestRoundtrip:
    def test_numpy_array(self, bus: DataBus):
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        bus.publish("test_np", arr)
        got = bus.latest("test_np")

        assert isinstance(got, np.ndarray)
        np.testing.assert_array_equal(got, arr)
        assert got.dtype == arr.dtype

    def test_dict_payload(self, bus: DataBus):
        data = {"joint1": 0.5, "joint2": -1.0}
        bus.publish("joints", data)
        got = bus.latest("joints")

        assert isinstance(got, dict)
        assert got == data

    def test_bytes_payload(self, bus: DataBus):
        raw = b"\xff\xfe\xfd"
        bus.publish("raw_ch", raw)
        got = bus.latest("raw_ch")

        assert isinstance(got, bytes)
        assert got == raw

    def test_empty_bytes(self, bus: DataBus):
        bus.publish("empty", b"")
        got = bus.latest("empty")
        assert got == b""

    def test_latest_missing_channel_returns_none(self, bus: DataBus):
        assert bus.latest("nonexistent", timeout_s=0.05) is None


# ── subscribe ────────────────────────────────────────────────────────


class TestSubscribe:
    def test_callback_receives_decoded_data(self, bus: DataBus, backend):
        received: list[Any] = []
        event = threading.Event()

        def on_data(data):
            received.append(data)
            event.set()

        sub = bus.subscribe("sub_test", on_data, policy="latest")
        try:
            time.sleep(0.05)
            bus.publish("sub_test", {"hello": "world"})
            assert event.wait(timeout=2.0), "Callback was not invoked"
            assert len(received) >= 1
            assert received[0] == {"hello": "world"}
        finally:
            sub.close()

    def test_raw_subscribe_receives_sample(self, bus: DataBus, backend):
        received: list[Sample] = []
        event = threading.Event()

        def on_raw(sample: Sample):
            received.append(sample)
            event.set()

        sub = bus.subscribe("raw_sub", on_raw, policy="latest", raw=True)
        try:
            time.sleep(0.05)
            bus.publish("raw_sub", b"\x01\x02")
            assert event.wait(timeout=2.0), "Raw callback was not invoked"
            assert len(received) >= 1
            assert isinstance(received[0], Sample)
        finally:
            sub.close()

    def test_unsubscribe(self, bus: DataBus, backend):
        received: list[Any] = []

        def on_data(data):
            received.append(data)

        sub = bus.subscribe("unsub_test", on_data, policy="latest")
        # Close *before* publishing so the poll loop no longer delivers.
        # The sleep ensures the backend has fully deregistered the watcher.
        sub.close()
        time.sleep(0.05)
        bus.publish("unsub_test", {"after": "close"})
        time.sleep(0.15)
        assert len(received) == 0


# ── HeaderTemplate caching ───────────────────────────────────────────


class TestTemplateCaching:
    def test_template_reused(self, bus: DataBus):
        arr = np.zeros((10,), dtype=np.float32)
        bus.publish("cache_ch", arr)
        tmpl1 = bus._templates.get("cache_ch")
        bus.publish("cache_ch", arr)
        tmpl2 = bus._templates.get("cache_ch")
        assert tmpl1 is tmpl2

    def test_template_invalidated_on_shape_change(self, bus: DataBus):
        arr1 = np.zeros((10,), dtype=np.float32)
        bus.publish("shape_ch", arr1)
        tmpl1 = bus._templates.get("shape_ch")

        arr2 = np.zeros((20,), dtype=np.float32)
        bus.publish("shape_ch", arr2)
        tmpl2 = bus._templates.get("shape_ch")

        assert tmpl1 is not tmpl2

    def test_seq_increments(self, bus: DataBus, backend):
        bus.publish("seq_ch", b"a")
        bus.publish("seq_ch", b"b")
        bus.publish("seq_ch", b"c")
        tmpl = bus._templates["seq_ch"]
        assert tmpl.seq == 3


# ── error handling ───────────────────────────────────────────────────


class TestErrors:
    def test_unsupported_type_raises(self, bus: DataBus):
        with pytest.raises(TypeError, match="Unsupported sample type"):
            bus.publish("bad", 42)  # type: ignore[arg-type]


# ── context manager ──────────────────────────────────────────────────


class TestLatestRaw:
    def test_raw_latest_returns_sample(self, bus: DataBus):
        bus.publish("raw_latest", b"\xaa\xbb")
        got = bus.latest("raw_latest", raw=True)
        assert isinstance(got, Sample)
        assert len(got.payload) > 0

    def test_raw_latest_missing_returns_none(self, bus: DataBus):
        got = bus.latest("no_such_ch", timeout_s=0.05, raw=True)
        assert got is None


class TestNumpyWritable:
    def test_decoded_array_is_writable(self, bus: DataBus):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        bus.publish("writable", arr)
        got = bus.latest("writable")
        assert isinstance(got, np.ndarray)
        got[0] = 99.0
        assert got[0] == 99.0


class TestNegativeMaxAge:
    def test_negative_max_age_raises(self, bus: DataBus):
        bus.publish("neg_age", b"\x01")
        with pytest.raises(ValueError, match="max_age_ms must be >= 0"):
            bus.latest("neg_age", max_age_ms=-1)


# ── context manager ──────────────────────────────────────────────────


class TestContextManager:
    def test_context_manager(self, tmp_path):
        be = FilesystemBackend(base_dir=tmp_path)
        with DataBus(be, TWIN_UUID) as bus:
            bus.publish("cm_ch", b"data")
