"""Unit tests for the camera frame-filter helper (``cyberwave.driver.sensors.frame_filter``).

The driver can optionally consult a Zenoh channel for a transformed
("anonymised") frame and substitute it into the WebRTC stream before
encoding. These tests cover the meaningful states:

* filter disabled (legacy pass-through)
* fresh, shape-matched processed frame -> substitute
* stale -> emit a black frame (privacy-safe default)
* shape / dtype mismatch -> emit a black frame (fail-closed)
* stale-log rate limiting
* non-ndarray samples are dropped

The tests target the standalone :class:`FrameFilter` helper directly,
which keeps them free of the driver lifecycle. The wiring (subscribe +
in-place substitution) is covered in ``test_driver_camera_mixin.py``.
"""

from __future__ import annotations

import logging

import numpy as np

from cyberwave.driver.sensors.frame_filter import (
    FrameFilter,
    ProcessedFrameSlot,
)


def _frame(value: int = 200) -> np.ndarray:
    return np.full((30, 40, 3), value, dtype=np.uint8)


class TestFilterDisabled:
    """Without a configured channel ``apply`` returns ``None`` (pass-through)."""

    def test_apply_returns_none(self):
        f = FrameFilter(channel=None)
        assert f.apply(_frame()) is None

    def test_enabled_property_reflects_channel(self):
        assert FrameFilter(channel=None).enabled is False
        assert FrameFilter(channel="frames/filtered").enabled is True


class TestFreshProcessedFrame:
    def test_returns_processed_frame_when_fresh_and_matched(self):
        f = FrameFilter(channel="frames/filtered", freshness_ms=200)
        processed = _frame(value=42)
        f.slot.put(processed)

        out = f.apply(_frame())

        assert out is processed

    def test_blank_frame_when_no_processed_frame_arrived_yet(self):
        # Worker hasn't published anything yet -> blank frame.
        f = FrameFilter(channel="frames/filtered")

        out = f.apply(_frame())

        assert out is not None
        assert out.shape == (30, 40, 3)
        assert (out == 0).all()


class TestStaleProcessedFrame:
    def test_blank_frame_when_stale(self):
        f = FrameFilter(channel="frames/filtered", freshness_ms=100)
        f.slot.put(_frame(value=99))
        # Rewind the slot's timestamp so the cached frame counts as stale.
        f.slot.timestamp -= 1.0

        out = f.apply(_frame(value=200))
        assert out is not None
        assert (out == 0).all()


class TestShapeAndDtypeMismatch:
    """Fail-closed when the worker publishes a wrongly-shaped frame."""

    def test_blank_frame_when_shape_mismatch(self):
        f = FrameFilter(channel="frames/filtered")
        # Worker published a 60x80 frame but the camera emits 30x40.
        bad = np.zeros((60, 80, 3), dtype=np.uint8)
        f.slot.put(bad)

        out = f.apply(_frame())

        assert out is not None
        assert out.shape == (30, 40, 3)
        assert (out == 0).all()

    def test_blank_frame_when_dtype_mismatch(self):
        f = FrameFilter(channel="frames/filtered")
        # Worker normalised to float32 by mistake.
        bad = np.zeros((30, 40, 3), dtype=np.float32)
        f.slot.put(bad)

        out = f.apply(_frame())

        assert out is not None
        assert out.dtype == np.uint8
        assert (out == 0).all()

    def test_mismatch_is_logged_with_diagnostics(self, caplog):
        f = FrameFilter(channel="frames/filtered", stale_log_s=1.0)
        f.slot.put(np.zeros((60, 80, 3), dtype=np.uint8))
        # Pre-expire the window so the next apply flushes its summary.
        f._stats.start_ts -= 10_000.0
        with caplog.at_level(logging.WARNING, logger="cyberwave.driver.sensors.frame_filter"):
            f.apply(_frame())
        assert any("shape_mismatch" in r.message for r in caplog.records), [
            r.message for r in caplog.records
        ]


class TestBlankFrameSummaryLog:
    """The window-expiry summary log replaces the old per-stale warning."""

    def test_no_log_until_window_expires(self, caplog):
        f = FrameFilter(channel="frames/filtered", freshness_ms=10, stale_log_s=60.0)
        with caplog.at_level(logging.WARNING, logger="cyberwave.driver.sensors.frame_filter"):
            for _ in range(5):
                f.apply(_frame())
        stale_logs = [r for r in caplog.records if "[FRAME_FILTER]" in r.message]
        assert stale_logs == [], [r.message for r in caplog.records]

    def test_one_summary_per_window(self, caplog):
        f = FrameFilter(channel="frames/filtered", freshness_ms=10, stale_log_s=30.0)
        with caplog.at_level(logging.WARNING, logger="cyberwave.driver.sensors.frame_filter"):
            for _ in range(3):
                f.apply(_frame())
            # Force the window to expire; the next apply flushes one summary.
            f._stats.start_ts -= 10_000.0
            f.apply(_frame())
            # Subsequent applies in the new window should NOT re-log.
            for _ in range(3):
                f.apply(_frame())
        stale_logs = [r for r in caplog.records if "[FRAME_FILTER]" in r.message]
        assert len(stale_logs) == 1, [r.message for r in caplog.records]

    def test_emits_again_after_window_elapses(self, caplog):
        f = FrameFilter(channel="frames/filtered", freshness_ms=10, stale_log_s=30.0)
        with caplog.at_level(logging.WARNING, logger="cyberwave.driver.sensors.frame_filter"):
            for _ in range(2):
                f._stats.start_ts -= 10_000.0
                f.apply(_frame())
        stale_logs = [r for r in caplog.records if "[FRAME_FILTER]" in r.message]
        assert len(stale_logs) == 2, [r.message for r in caplog.records]

    def test_no_log_when_no_blanks_in_window(self, caplog):
        # Worker is healthy: every apply gets a fresh, well-formed frame.
        f = FrameFilter(channel="frames/filtered", freshness_ms=200, stale_log_s=30.0)
        with caplog.at_level(logging.WARNING, logger="cyberwave.driver.sensors.frame_filter"):
            for _ in range(5):
                f.slot.put(_frame(value=42))
                f.apply(_frame())
            f._stats.start_ts -= 10_000.0
            f.slot.put(_frame(value=42))
            f.apply(_frame())
        stale_logs = [r for r in caplog.records if "[FRAME_FILTER]" in r.message]
        assert stale_logs == [], [r.message for r in caplog.records]

    def test_summary_includes_percentage_and_count(self, caplog):
        f = FrameFilter(channel="frames/filtered", freshness_ms=200, stale_log_s=30.0)
        with caplog.at_level(logging.WARNING, logger="cyberwave.driver.sensors.frame_filter"):
            # 1 healthy frame, then 9 with a forced-stale slot.
            f.slot.put(_frame(value=42))
            f.apply(_frame())
            for _ in range(9):
                f.slot.timestamp -= 10.0
                f.apply(_frame())
            # Force the window to expire and flush a summary on the next apply.
            f._stats.start_ts -= 10_000.0
            f.apply(_frame())
        flush_records = [r for r in caplog.records if "[FRAME_FILTER]" in r.message]
        assert flush_records, "expected at least one [FRAME_FILTER] summary log"
        msg = flush_records[-1].message
        assert "%" in msg
        assert "stale=" in msg
        assert "frames in last" in msg

    def test_high_blank_rate_uses_dead_worker_advice(self, caplog):
        f = FrameFilter(channel="frames/filtered", freshness_ms=10, stale_log_s=30.0)
        with caplog.at_level(logging.WARNING, logger="cyberwave.driver.sensors.frame_filter"):
            for _ in range(10):
                f.apply(_frame())
            f._stats.start_ts -= 10_000.0
            f.apply(_frame())
        stale_logs = [r for r in caplog.records if "[FRAME_FILTER]" in r.message]
        assert stale_logs, [r.message for r in caplog.records]
        assert "Worker likely down" in stale_logs[-1].message

    def test_low_blank_rate_uses_freshness_advice(self, caplog):
        f = FrameFilter(channel="frames/filtered", freshness_ms=200, stale_log_s=30.0)
        with caplog.at_level(logging.WARNING, logger="cyberwave.driver.sensors.frame_filter"):
            # 99 healthy + 1 stale = 1% blank.
            for _ in range(99):
                f.slot.put(_frame(value=42))
                f.apply(_frame())
            f.slot.timestamp -= 10.0  # force the next read to be stale
            f.apply(_frame())
            f._stats.start_ts -= 10_000.0
            f.slot.put(_frame(value=42))
            f.apply(_frame())  # flush
        stale_logs = [r for r in caplog.records if "[FRAME_FILTER]" in r.message]
        assert stale_logs, [r.message for r in caplog.records]
        assert "freshness window too tight" in stale_logs[-1].message


class TestProcessedFrameDispatch:
    def test_ignores_non_ndarray_samples(self):
        f = FrameFilter(channel="frames/filtered")
        f.store_processed({"hello": "world"})
        assert f.slot.get_if_fresh(60.0) is None

    def test_stores_ndarray_samples(self):
        f = FrameFilter(channel="frames/filtered")
        sample = _frame(value=7)
        f.store_processed(sample)
        assert f.slot.get_if_fresh(60.0) is sample


class TestProcessedFrameSlot:
    def test_empty_slot_returns_none(self):
        slot = ProcessedFrameSlot()
        assert slot.get_if_fresh(60.0) is None

    def test_returns_frame_within_max_age(self):
        slot = ProcessedFrameSlot()
        sample = _frame(value=123)
        slot.put(sample)
        assert slot.get_if_fresh(60.0) is sample

    def test_returns_none_when_stale(self):
        slot = ProcessedFrameSlot()
        slot.put(_frame())
        slot.timestamp -= 10.0
        assert slot.get_if_fresh(0.5) is None


class TestConstructorClamping:
    def test_negative_freshness_clamped_to_zero(self):
        f = FrameFilter(channel="frames/filtered", freshness_ms=-100)
        assert f.freshness_s == 0.0

    def test_negative_stale_log_clamped_disables_logging(self, caplog):
        f = FrameFilter(
            channel="frames/filtered",
            freshness_ms=10,
            stale_log_s=-1.0,
        )
        with caplog.at_level(logging.WARNING, logger="cyberwave.driver.sensors.frame_filter"):
            f.apply(_frame())
        stale_logs = [r for r in caplog.records if "[FRAME_FILTER]" in r.message]
        assert stale_logs == []
