"""Tests for TimeIndexedRingBuffer — insertion, eviction, binary search, thread safety."""

from __future__ import annotations

import threading
import time

import pytest

from cyberwave.data.ring_buffer import (
    BracketResult,
    TimeIndexedRingBuffer,
    TimestampedSample,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_capacity(self) -> None:
        buf = TimeIndexedRingBuffer()
        assert buf.capacity == 1000

    def test_custom_capacity(self) -> None:
        buf = TimeIndexedRingBuffer(capacity=50)
        assert buf.capacity == 50

    def test_invalid_capacity_zero(self) -> None:
        with pytest.raises(ValueError, match="capacity must be >= 1"):
            TimeIndexedRingBuffer(capacity=0)

    def test_invalid_capacity_negative(self) -> None:
        with pytest.raises(ValueError, match="capacity must be >= 1"):
            TimeIndexedRingBuffer(capacity=-5)

    def test_empty_on_creation(self) -> None:
        buf = TimeIndexedRingBuffer()
        assert len(buf) == 0
        assert buf.empty
        assert buf.oldest_ts is None
        assert buf.newest_ts is None


# ---------------------------------------------------------------------------
# Insertion and eviction
# ---------------------------------------------------------------------------


class TestInsertionAndEviction:
    def test_append_single(self) -> None:
        buf: TimeIndexedRingBuffer[str] = TimeIndexedRingBuffer(capacity=10)
        buf.append(1.0, "a")
        assert len(buf) == 1
        assert not buf.empty
        assert buf.oldest_ts == 1.0
        assert buf.newest_ts == 1.0

    def test_append_multiple_in_order(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer(capacity=10)
        for i in range(5):
            buf.append(float(i), i)
        assert len(buf) == 5
        assert buf.oldest_ts == 0.0
        assert buf.newest_ts == 4.0

    def test_append_equal_timestamps(self) -> None:
        buf: TimeIndexedRingBuffer[str] = TimeIndexedRingBuffer(capacity=10)
        buf.append(1.0, "a")
        buf.append(1.0, "b")
        assert len(buf) == 2

    def test_append_out_of_order_raises(self) -> None:
        buf: TimeIndexedRingBuffer[str] = TimeIndexedRingBuffer(capacity=10)
        buf.append(2.0, "a")
        with pytest.raises(ValueError, match="Out-of-order insert"):
            buf.append(1.0, "b")

    def test_eviction_at_capacity(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer(capacity=3)
        buf.append(1.0, 10)
        buf.append(2.0, 20)
        buf.append(3.0, 30)
        assert len(buf) == 3
        assert buf.oldest_ts == 1.0

        buf.append(4.0, 40)
        assert len(buf) == 3
        assert buf.oldest_ts == 2.0
        assert buf.newest_ts == 4.0

    def test_eviction_preserves_order(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer(capacity=3)
        for i in range(10):
            buf.append(float(i), i)

        samples = buf.all_samples()
        assert len(samples) == 3
        assert samples[0].ts == 7.0
        assert samples[1].ts == 8.0
        assert samples[2].ts == 9.0

    def test_capacity_one(self) -> None:
        buf: TimeIndexedRingBuffer[str] = TimeIndexedRingBuffer(capacity=1)
        buf.append(1.0, "a")
        assert len(buf) == 1
        buf.append(2.0, "b")
        assert len(buf) == 1
        assert buf.newest_ts == 2.0
        s = buf.latest()
        assert s is not None
        assert s.value == "b"


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_empty_buffer(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        buf.clear()
        assert len(buf) == 0

    def test_clear_populated_buffer(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        for i in range(10):
            buf.append(float(i), i)
        buf.clear()
        assert len(buf) == 0
        assert buf.empty
        assert buf.oldest_ts is None
        assert buf.newest_ts is None


# ---------------------------------------------------------------------------
# Binary search — at()
# ---------------------------------------------------------------------------


class TestAt:
    def test_empty_buffer(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        result = buf.at(1.0)
        assert result.before is None
        assert result.after is None
        assert result.exact is None

    def test_exact_match(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        buf.append(1.0, 10)
        buf.append(2.0, 20)
        buf.append(3.0, 30)

        result = buf.at(2.0)
        assert result.exact is not None
        assert result.exact.ts == 2.0
        assert result.exact.value == 20

    def test_between_samples(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        buf.append(1.0, 10)
        buf.append(3.0, 30)

        result = buf.at(2.0)
        assert result.exact is None
        assert result.before is not None
        assert result.after is not None
        assert result.before.ts == 1.0
        assert result.before.value == 10
        assert result.after.ts == 3.0
        assert result.after.value == 30

    def test_before_all_samples(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        buf.append(5.0, 50)
        buf.append(6.0, 60)

        result = buf.at(1.0)
        assert result.exact is None
        assert result.before is None
        assert result.after is not None
        assert result.after.ts == 5.0

    def test_after_all_samples(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        buf.append(1.0, 10)
        buf.append(2.0, 20)

        result = buf.at(5.0)
        assert result.exact is None
        assert result.before is not None
        assert result.before.ts == 2.0
        assert result.after is None

    def test_exact_match_first_element(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        buf.append(1.0, 10)
        buf.append(2.0, 20)

        result = buf.at(1.0)
        assert result.exact is not None
        assert result.exact.value == 10

    def test_exact_match_last_element(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        buf.append(1.0, 10)
        buf.append(2.0, 20)

        result = buf.at(2.0)
        assert result.exact is not None
        assert result.exact.value == 20

    def test_single_element_exact(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        buf.append(5.0, 50)

        result = buf.at(5.0)
        assert result.exact is not None
        assert result.exact.value == 50

    def test_single_element_before(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        buf.append(5.0, 50)

        result = buf.at(3.0)
        assert result.before is None
        assert result.after is not None
        assert result.after.ts == 5.0

    def test_single_element_after(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        buf.append(5.0, 50)

        result = buf.at(7.0)
        assert result.before is not None
        assert result.before.ts == 5.0
        assert result.after is None


# ---------------------------------------------------------------------------
# Range query — window()
# ---------------------------------------------------------------------------


class TestWindow:
    def test_empty_buffer(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        result = buf.window(0.0, 10.0)
        assert result == []

    def test_all_samples_in_range(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        for i in range(5):
            buf.append(float(i), i * 10)
        result = buf.window(0.0, 4.0)
        assert len(result) == 5
        assert result[0].ts == 0.0
        assert result[-1].ts == 4.0

    def test_subset_of_samples(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        for i in range(10):
            buf.append(float(i), i * 10)
        result = buf.window(3.0, 6.0)
        assert len(result) == 4
        assert [s.ts for s in result] == [3.0, 4.0, 5.0, 6.0]

    def test_inclusive_boundaries(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        buf.append(1.0, 10)
        buf.append(2.0, 20)
        buf.append(3.0, 30)

        result = buf.window(1.0, 3.0)
        assert len(result) == 3

    def test_no_samples_in_range(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        buf.append(1.0, 10)
        buf.append(2.0, 20)

        result = buf.window(5.0, 10.0)
        assert result == []

    def test_range_between_samples(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        buf.append(1.0, 10)
        buf.append(5.0, 50)

        result = buf.window(2.0, 4.0)
        assert result == []

    def test_from_equals_to(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        buf.append(1.0, 10)
        buf.append(2.0, 20)
        buf.append(3.0, 30)

        result = buf.window(2.0, 2.0)
        assert len(result) == 1
        assert result[0].ts == 2.0

    def test_invalid_range(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        with pytest.raises(ValueError, match="from_t.*must be <= to_t"):
            buf.window(5.0, 1.0)


# ---------------------------------------------------------------------------
# latest() and all_samples()
# ---------------------------------------------------------------------------


class TestLatestAndAllSamples:
    def test_latest_empty(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        assert buf.latest() is None

    def test_latest_returns_newest(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        buf.append(1.0, 10)
        buf.append(2.0, 20)
        s = buf.latest()
        assert s is not None
        assert s.ts == 2.0
        assert s.value == 20

    def test_all_samples_empty(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        assert buf.all_samples() == []

    def test_all_samples_returns_copy(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        buf.append(1.0, 10)
        buf.append(2.0, 20)
        samples = buf.all_samples()
        assert len(samples) == 2
        buf.append(3.0, 30)
        assert len(samples) == 2


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_append_and_read(self) -> None:
        """Multiple writers + readers should not corrupt internal state."""
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer(capacity=100)
        errors: list[Exception] = []
        stop = threading.Event()

        def writer(start_ts: float) -> None:
            try:
                for i in range(50):
                    buf.append(start_ts + i * 0.001, i)
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                while not stop.is_set():
                    buf.at(time.time())
                    buf.window(0.0, time.time())
                    len(buf)
            except Exception as e:
                errors.append(e)

        reader_thread = threading.Thread(target=reader, daemon=True)
        reader_thread.start()

        t0 = time.time()
        writer(t0)

        stop.set()
        reader_thread.join(timeout=2.0)

        assert not errors, f"Thread errors: {errors}"

    def test_concurrent_multi_writer(self) -> None:
        """Two writers racing on the same buffer must not corrupt state.

        Some appends may raise ValueError (out-of-order) — that's expected.
        The invariant is: no unhandled exceptions and timestamps in the
        buffer remain non-decreasing.
        """
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer(capacity=200)
        errors: list[Exception] = []

        def writer(writer_id: int, base_ts: float) -> None:
            for i in range(100):
                try:
                    buf.append(base_ts + i * 0.0001, writer_id * 1000 + i)
                except ValueError:
                    pass
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=writer, args=(0, 0.0))
        t2 = threading.Thread(target=writer, args=(1, 0.0))
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert not errors, f"Unexpected errors: {errors}"

        samples = buf.all_samples()
        for i in range(1, len(samples)):
            assert samples[i].ts >= samples[i - 1].ts, (
                f"Non-decreasing invariant violated at index {i}: "
                f"{samples[i - 1].ts} > {samples[i].ts}"
            )

    def test_concurrent_reads_do_not_block(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer()
        for i in range(100):
            buf.append(float(i), i)

        results: list[int] = []

        def reader() -> None:
            for _ in range(50):
                r = buf.at(50.0)
                if r.exact is not None:
                    results.append(r.exact.value)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(results) == 200
        assert all(v == 50 for v in results)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_empty(self) -> None:
        buf = TimeIndexedRingBuffer(capacity=42)
        assert "capacity=42" in repr(buf)
        assert "size=0" in repr(buf)

    def test_repr_populated(self) -> None:
        buf: TimeIndexedRingBuffer[int] = TimeIndexedRingBuffer(capacity=10)
        for i in range(5):
            buf.append(float(i), i)
        assert "size=5" in repr(buf)


# ---------------------------------------------------------------------------
# TimestampedSample and BracketResult dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_timestamped_sample_frozen(self) -> None:
        s = TimestampedSample(ts=1.0, value="hello")
        with pytest.raises(AttributeError):
            s.ts = 2.0  # type: ignore[misc]

    def test_bracket_result_defaults(self) -> None:
        r = BracketResult()
        assert r.before is None
        assert r.after is None
        assert r.exact is None

    def test_bracket_result_with_values(self) -> None:
        s1 = TimestampedSample(ts=1.0, value=10)
        s2 = TimestampedSample(ts=2.0, value=20)
        r = BracketResult(before=s1, after=s2)
        assert r.before is s1
        assert r.after is s2
        assert r.exact is None
