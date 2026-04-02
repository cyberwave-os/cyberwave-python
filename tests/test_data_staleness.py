"""Tests for max_age_ms staleness on DataBus.latest() — Phase 1 temporal sync."""

from __future__ import annotations

import time

import numpy as np
import pytest

from cyberwave.data.api import DataBus
from cyberwave.data.filesystem_backend import FilesystemBackend

TWIN_UUID = "550e8400-e29b-41d4-a716-446655440000"


@pytest.fixture()
def backend(tmp_path):
    be = FilesystemBackend(base_dir=tmp_path, poll_interval_s=0.02)
    yield be
    be.close()


@pytest.fixture()
def bus(backend):
    return DataBus(backend, TWIN_UUID, key_prefix="cw")


class TestMaxAgeMs:
    def test_fresh_sample_returned(self, bus: DataBus):
        arr = np.array([1, 2, 3], dtype=np.int32)
        bus.publish("fresh", arr)
        got = bus.latest("fresh", max_age_ms=5000)
        assert got is not None
        np.testing.assert_array_equal(got, arr)

    def test_stale_sample_returns_none(self, bus: DataBus):
        arr = np.array([1, 2, 3], dtype=np.int32)
        bus.publish("stale", arr)
        time.sleep(0.15)
        got = bus.latest("stale", max_age_ms=50)
        assert got is None

    def test_no_max_age_always_returns(self, bus: DataBus):
        arr = np.array([1], dtype=np.float64)
        bus.publish("always", arr)
        time.sleep(0.15)
        got = bus.latest("always")
        assert got is not None

    def test_dict_staleness(self, bus: DataBus):
        bus.publish("dict_stale", {"val": 42})
        time.sleep(0.15)
        assert bus.latest("dict_stale", max_age_ms=50) is None

    def test_dict_fresh(self, bus: DataBus):
        bus.publish("dict_fresh", {"val": 42})
        got = bus.latest("dict_fresh", max_age_ms=5000)
        assert got == {"val": 42}

    def test_bytes_staleness(self, bus: DataBus):
        bus.publish("bytes_stale", b"\x00")
        time.sleep(0.15)
        assert bus.latest("bytes_stale", max_age_ms=50) is None

    def test_zero_max_age_returns_none_for_any_sample(self, bus: DataBus):
        bus.publish("zero_age", b"\x01")
        time.sleep(0.01)
        got = bus.latest("zero_age", max_age_ms=0)
        assert got is None

    def test_large_max_age_returns_sample(self, bus: DataBus):
        bus.publish("big_age", b"\x02")
        got = bus.latest("big_age", max_age_ms=999_999)
        assert got is not None

    def test_negative_max_age_raises(self, bus: DataBus):
        bus.publish("neg", b"\x03")
        with pytest.raises(ValueError, match="max_age_ms must be >= 0"):
            bus.latest("neg", max_age_ms=-100)
