"""Tests for the ``CV2VideoTrack.frame_callback`` return-value contract.

Backward-compatible contract:
- ``callback(frame, count) -> None`` keeps the original frame (legacy).
- ``callback(frame, count) -> ndarray`` replaces the frame before encoding.
- Incompatible return values are logged and the original frame is kept.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("cv2", reason="OpenCV not installed")
pytest.importorskip("av", reason="pyav not installed")

from cyberwave.sensor.camera_cv2 import CV2VideoTrack  # noqa: E402


def _patched_track(
    callback,
    *,
    frame: np.ndarray | None = None,
) -> tuple[CV2VideoTrack, np.ndarray]:
    """Construct a CV2VideoTrack without opening a real device.

    We bypass __init__'s capture-opening logic and assemble the bare minimum
    state needed for ``recv()`` to exercise the frame-callback path.
    """
    if frame is None:
        frame = np.full((30, 40, 3), 200, dtype=np.uint8)

    track = CV2VideoTrack.__new__(CV2VideoTrack)
    # Replicate aiortc's MediaStreamTrack base __init__ side-effects manually
    # (just enough for the recv path; we never publish frames).
    track.__class__.__bases__[0].__init__(track)
    track.cap = MagicMock()
    track.cap.read.return_value = (True, frame.copy())
    track.frame_count = 0
    track.frame_0_timestamp = 0
    track.frame_0_timestamp_monotonic = 0
    track.time_reference = None
    track.frame_callback = callback
    track.keyframe_interval = None
    track._frames_since_keyframe = 0
    track._capture_overruns = 0
    track._last_warned_overruns = 0
    track.fps = 30
    track.actual_fps = 30
    track._frame_callback_warn_count = 0
    return track, frame


def _capture_videoframe_input(monkeypatch) -> list[np.ndarray]:
    """Capture every ndarray VideoFrame.from_ndarray sees, return that list."""
    captured: list[np.ndarray] = []

    real = MagicMock()

    def fake_from_ndarray(arr, format):  # noqa: A002 - mirror aiortc API
        captured.append(arr.copy())
        vf = MagicMock()
        vf.pts = 0
        # ``video_frame = video_frame.reformat(...)`` chains; return self.
        vf.reformat.return_value = vf
        return vf

    real.from_ndarray.side_effect = fake_from_ndarray
    monkeypatch.setattr("cyberwave.sensor.camera_cv2.VideoFrame", real)
    return captured


def test_callback_returning_none_keeps_original_frame(monkeypatch):
    captured = _capture_videoframe_input(monkeypatch)
    seen: list[int] = []

    def cb(frame, count):
        seen.append(count)
        return None

    track, original = _patched_track(cb)
    # Patch _capture_timestamp so we don't need a TimeReference.
    track._capture_timestamp = lambda _ref: (0.0, 0.0)
    # Patch _normalize_frame to identity.
    track._normalize_frame = lambda f: f

    asyncio.run(track.recv())
    assert seen == [0]
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], original)


def test_callback_returning_ndarray_replaces_frame(monkeypatch):
    captured = _capture_videoframe_input(monkeypatch)

    replacement = np.zeros((30, 40, 3), dtype=np.uint8)

    def cb(_frame, _count):
        return replacement

    track, _ = _patched_track(cb)
    track._capture_timestamp = lambda _ref: (0.0, 0.0)
    track._normalize_frame = lambda f: f

    asyncio.run(track.recv())
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], replacement)


def test_callback_returning_garbage_keeps_original_frame(monkeypatch, caplog):
    captured = _capture_videoframe_input(monkeypatch)

    def cb(_frame, _count):
        return "not an array"  # incompatible return value

    track, original = _patched_track(cb)
    track._capture_timestamp = lambda _ref: (0.0, 0.0)
    track._normalize_frame = lambda f: f

    with caplog.at_level("WARNING", logger="cyberwave.sensor.camera_cv2"):
        asyncio.run(track.recv())
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], original)
    # Lock the warning contract so a future refactor can't silently swallow it.
    assert "incompatible value" in caplog.text
    assert track._frame_callback_warn_count == 1


def test_callback_raising_keeps_original_frame(monkeypatch):
    captured = _capture_videoframe_input(monkeypatch)

    def cb(_frame, _count):
        raise RuntimeError("boom")

    track, original = _patched_track(cb)
    track._capture_timestamp = lambda _ref: (0.0, 0.0)
    track._normalize_frame = lambda f: f

    asyncio.run(track.recv())
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], original)


def test_callback_returning_wrong_shape_keeps_original_frame(monkeypatch):
    captured = _capture_videoframe_input(monkeypatch)

    # Wrong number of channels (2 instead of 3): rejected.
    bad = np.zeros((30, 40, 2), dtype=np.uint8)

    def cb(_frame, _count):
        return bad

    track, original = _patched_track(cb)
    track._capture_timestamp = lambda _ref: (0.0, 0.0)
    track._normalize_frame = lambda f: f

    asyncio.run(track.recv())
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], original)


def test_callback_returning_wrong_height_keeps_original_frame(monkeypatch):
    """A callback that returns a resized frame must not silently reach the
    encoder — the WebRTC negotiated resolution depends on a stable shape."""
    captured = _capture_videoframe_input(monkeypatch)

    bad = np.zeros((60, 40, 3), dtype=np.uint8)  # 2x height

    def cb(_frame, _count):
        return bad

    track, original = _patched_track(cb)
    track._capture_timestamp = lambda _ref: (0.0, 0.0)
    track._normalize_frame = lambda f: f

    asyncio.run(track.recv())
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], original)


def test_callback_returning_wrong_dtype_keeps_original_frame(monkeypatch):
    """``av.VideoFrame.from_ndarray(format='bgr24')`` requires uint8 — float
    arrays would either crash the encoder or be silently truncated."""
    captured = _capture_videoframe_input(monkeypatch)

    bad = np.zeros((30, 40, 3), dtype=np.float32)

    def cb(_frame, _count):
        return bad

    track, original = _patched_track(cb)
    track._capture_timestamp = lambda _ref: (0.0, 0.0)
    track._normalize_frame = lambda f: f

    asyncio.run(track.recv())
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], original)


def test_warn_frame_callback_is_rate_limited(monkeypatch, caplog):
    """A chronically broken callback must not spam logs — only the first few
    occurrences plus every Nth thereafter should be emitted."""
    _capture_videoframe_input(monkeypatch)

    def cb(_frame, _count):
        return "garbage"

    track, _ = _patched_track(cb)
    track._capture_timestamp = lambda _ref: (0.0, 0.0)
    track._normalize_frame = lambda f: f

    with caplog.at_level("WARNING", logger="cyberwave.sensor.camera_cv2"):
        for _ in range(250):
            asyncio.run(track.recv())

    # Counter increments every frame...
    assert track._frame_callback_warn_count == 250
    # ...but logs are throttled: 1, 2, 100, 200 → 4 warnings, far fewer than 250.
    incompatible = [r for r in caplog.records if "incompatible value" in r.getMessage()]
    assert len(incompatible) == 4
