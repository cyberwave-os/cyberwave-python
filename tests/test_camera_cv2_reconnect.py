"""Tests for ``CV2VideoTrack`` source-side auto-reconnect.

When the upstream camera source blips (ffmpeg crash on the macOS host, USB
unplug, network flake on an IP camera), ``cap.read()`` starts returning
``False``. Without this behaviour the track returned ``None``, the aiortc
encoder asserted, and the WebRTC sender died — requiring a container restart
to recover. The contract tested here:

1. The first ``cap.read()`` failure logs an error and returns ``None`` only
   when no good frame has ever been captured (cold start with a dead source).
2. After a successful frame, subsequent failures fall back to the cached
   freeze frame so the encoder keeps a valid input and the stream freezes
   instead of dying.
3. After ``_RECONNECT_TRIGGER_FAILURES`` consecutive failures, a background
   reconnect task is scheduled (idempotent — re-entering ``recv()`` does not
   stack tasks).
4. When the reopen executor returns a fresh capture, the track swaps it in,
   resets the failure counter, and releases the old capture.
5. ``close()`` cancels the in-flight reconnect task.
6. Freeze ticks do not count as liveness (``captured_frame_count`` holds) and
   are paced to the configured frame interval.
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
    *,
    frame: np.ndarray | None = None,
) -> tuple[CV2VideoTrack, np.ndarray]:
    """Construct a CV2VideoTrack without opening a real device."""
    if frame is None:
        frame = np.full((30, 40, 3), 200, dtype=np.uint8)

    track = CV2VideoTrack.__new__(CV2VideoTrack)
    track.__class__.__bases__[0].__init__(track)
    track.cap = MagicMock()
    track.cap.read.return_value = (True, frame.copy())
    track.camera_id = "http://127.0.0.1:8554/camera.mjpg"
    track.frame_count = 0
    track.frame_0_timestamp = 0
    track.frame_0_timestamp_monotonic = 0
    track.time_reference = None
    track.frame_callback = None
    track.keyframe_interval = None
    track._frames_since_keyframe = 0
    track.fps = 30
    track.actual_fps = 30
    track._frame_callback_warn_count = 0
    track._consecutive_read_failures = 0
    track._last_good_frame_bgr = None
    track._reconnect_task = None
    track.captured_frame_count = 0
    track._negotiated_fourcc_ascii = None
    track._capture_timestamp = lambda _ref: (0.0, 0.0)
    track._normalize_frame = lambda f: f
    track._store_frame_metadata_for_sync = lambda **_kwargs: None
    track._capture_sync_frame = lambda *_args, **_kwargs: None
    return track, frame


def _patch_asyncio_sleep(monkeypatch) -> None:
    """Replace ``asyncio.sleep`` inside camera_cv2 with a zero-delay yield.

    Captures the real ``asyncio.sleep`` first so the replacement does not
    recursively call itself when the module attribute is rebound.
    """
    real_sleep = asyncio.sleep

    async def _yield(_seconds):
        await real_sleep(0)

    monkeypatch.setattr("cyberwave.sensor.camera_cv2.asyncio.sleep", _yield)


def _stub_videoframe(monkeypatch) -> list[np.ndarray]:
    captured: list[np.ndarray] = []
    real = MagicMock()

    def fake_from_ndarray(arr, format):  # noqa: A002
        captured.append(arr.copy())
        vf = MagicMock()
        vf.pts = 0
        vf.reformat.return_value = vf
        return vf

    real.from_ndarray.side_effect = fake_from_ndarray
    monkeypatch.setattr("cyberwave.sensor.camera_cv2.VideoFrame", real)
    return captured


def test_first_read_failure_with_no_cached_frame_returns_none(monkeypatch, caplog):
    """Cold start with a dead source: legacy behaviour (None + error log)."""
    _stub_videoframe(monkeypatch)
    track, _ = _patched_track()
    track.cap.read.return_value = (False, None)

    with caplog.at_level("ERROR", logger="cyberwave.sensor.camera_cv2"):
        result = asyncio.run(track.recv())

    assert result is None
    assert track._consecutive_read_failures == 1
    assert track._last_good_frame_bgr is None
    assert "no cached frame" in caplog.text.lower()


def test_failure_after_success_returns_freeze_frame(monkeypatch):
    """A cached frame must be reused so aiortc doesn't see ``None``."""
    captured = _stub_videoframe(monkeypatch)
    track, original = _patched_track()

    asyncio.run(track.recv())
    assert track._last_good_frame_bgr is not None
    np.testing.assert_array_equal(track._last_good_frame_bgr, original)

    track.cap.read.return_value = (False, None)
    result = asyncio.run(track.recv())

    assert result is not None
    assert track._consecutive_read_failures == 1
    assert len(captured) == 2
    np.testing.assert_array_equal(captured[1], original)


def test_threshold_failures_schedule_reconnect(monkeypatch):
    """After N failures the reconnect task is scheduled exactly once."""
    _stub_videoframe(monkeypatch)
    track, _ = _patched_track()

    asyncio.run(track.recv())
    track.cap.read.return_value = (False, None)

    # Block the reconnect coroutine indefinitely so we can verify the task
    # is alive at the moment of inspection (asyncio.run tears down all tasks
    # on exit, so all assertions about task state must happen before then).
    started = asyncio.Event()
    spawn_count = 0

    async def fake_loop():
        nonlocal spawn_count
        spawn_count += 1
        started.set()
        await asyncio.Event().wait()

    track._reconnect_loop = fake_loop  # type: ignore[assignment]

    async def drive():
        for _ in range(CV2VideoTrack._RECONNECT_TRIGGER_FAILURES):
            await track.recv()
        # One extra tick must be a no-op (idempotent scheduling).
        await track.recv()
        await started.wait()
        # Inspect state while the loop is still alive.
        task = track._reconnect_task
        assert task is not None
        assert not task.done()
        # Cancel the long-lived fake_loop so asyncio.run can shut down cleanly.
        task.cancel()

    asyncio.run(drive())
    assert spawn_count == 1
    assert track._consecutive_read_failures >= CV2VideoTrack._RECONNECT_TRIGGER_FAILURES


def test_reconnect_swaps_capture_and_releases_old(monkeypatch):
    """A successful reopen replaces ``self.cap`` and releases the old one."""
    _stub_videoframe(monkeypatch)
    track, _ = _patched_track()
    asyncio.run(track.recv())

    old_cap = track.cap
    new_cap = MagicMock()
    new_cap.read.return_value = (True, np.full((30, 40, 3), 100, dtype=np.uint8))

    track._reopen_capture_blocking = lambda: new_cap  # type: ignore[assignment]
    _patch_asyncio_sleep(monkeypatch)

    track._consecutive_read_failures = CV2VideoTrack._RECONNECT_TRIGGER_FAILURES

    asyncio.run(track._reconnect_loop())

    assert track.cap is new_cap
    assert track._consecutive_read_failures == 0
    old_cap.release.assert_called_once()


def test_reconnect_retries_until_success(monkeypatch):
    """A failing reopen is retried (backoff capped at the configured max)."""
    _stub_videoframe(monkeypatch)
    track, _ = _patched_track()
    asyncio.run(track.recv())

    new_cap = MagicMock()
    new_cap.read.return_value = (True, np.zeros((30, 40, 3), dtype=np.uint8))
    attempts: list[int] = []

    def reopen():
        attempts.append(1)
        return None if len(attempts) < 3 else new_cap

    track._reopen_capture_blocking = reopen  # type: ignore[assignment]
    _patch_asyncio_sleep(monkeypatch)

    asyncio.run(track._reconnect_loop())

    assert len(attempts) == 3
    assert track.cap is new_cap


def test_reopen_then_immediate_failure_reaccumulates_and_can_rearm(monkeypatch):
    """Reopen success followed by an immediately-failing new cap must restart
    the failure counter against the new cap and re-arm the reconnect path.

    Realistic when the upstream source comes back up but isn't producing
    frames yet (e.g. ffmpeg restarted but has not grabbed the camera lock
    yet). Without this contract, a single transient flap could mask a
    continuing outage because the failure counter would never re-cross the
    threshold and a second reopen would never be scheduled.
    """
    _stub_videoframe(monkeypatch)
    track, _ = _patched_track()

    asyncio.run(track.recv())
    assert track._last_good_frame_bgr is not None

    new_cap = MagicMock()
    new_cap.read.return_value = (False, None)
    track._reopen_capture_blocking = lambda: new_cap  # type: ignore[assignment]
    _patch_asyncio_sleep(monkeypatch)

    track._consecutive_read_failures = CV2VideoTrack._RECONNECT_TRIGGER_FAILURES
    asyncio.run(track._reconnect_loop())

    assert track.cap is new_cap
    assert track._consecutive_read_failures == 0

    # Drive ``recv()`` until the threshold is crossed against the new cap.
    # Stub ``_maybe_schedule_reconnect`` to just observe — we want to verify
    # the gate fires again, without spawning a real task that races teardown.
    schedule_calls: list[int] = []

    def counted_schedule():
        if track._consecutive_read_failures >= CV2VideoTrack._RECONNECT_TRIGGER_FAILURES:
            schedule_calls.append(track._consecutive_read_failures)

    track._maybe_schedule_reconnect = counted_schedule  # type: ignore[assignment]

    for _ in range(CV2VideoTrack._RECONNECT_TRIGGER_FAILURES):
        result = asyncio.run(track.recv())
        assert result is not None  # freeze frame, never None after first success

    assert (
        track._consecutive_read_failures
        == CV2VideoTrack._RECONNECT_TRIGGER_FAILURES
    )
    assert schedule_calls, "gate should have re-fired after threshold re-cross"
    assert schedule_calls[-1] == CV2VideoTrack._RECONNECT_TRIGGER_FAILURES


def _record_sleeps(monkeypatch) -> list[float]:
    """Swap ``asyncio.sleep`` for a recorder that still yields to the loop."""
    real_sleep = asyncio.sleep
    seen: list[float] = []

    async def _recording(seconds):
        seen.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr("cyberwave.sensor.camera_cv2.asyncio.sleep", _recording)
    return seen


def test_captured_frame_count_ignores_freeze_frames(monkeypatch):
    """Liveness must track captures, not emitted frames.

    ``frame_count`` keeps climbing on freeze ticks, so ``captured_frame_count``
    has to freeze with the device for ``EdgeHealthCheck`` to go stale.
    """
    _stub_videoframe(monkeypatch)
    _record_sleeps(monkeypatch)
    track, _ = _patched_track()

    asyncio.run(track.recv())
    assert (track.frame_count, track.captured_frame_count) == (1, 1)

    track.cap.read.return_value = (False, None)
    for _ in range(5):
        assert asyncio.run(track.recv()) is not None  # freeze frames keep flowing

    # Emitted frames advanced (the WebRTC pipeline is still alive on purpose)
    # but the capture counter did not.
    assert track.frame_count == 6
    assert track.captured_frame_count == 1


def test_captured_frame_count_resumes_after_blip(monkeypatch):
    """A blip shorter than ``stale_timeout`` must not cost the stream its health.

    The freeze fallback exists for exactly this case (ffmpeg restart,
    IP-camera flake).
    """
    _stub_videoframe(monkeypatch)
    _record_sleeps(monkeypatch)
    track, original = _patched_track()

    asyncio.run(track.recv())
    track.cap.read.return_value = (False, None)
    asyncio.run(track.recv())
    asyncio.run(track.recv())
    assert track.captured_frame_count == 1

    track.cap.read.return_value = (True, original.copy())
    asyncio.run(track.recv())

    assert track.captured_frame_count == 2
    assert track._consecutive_read_failures == 0


def test_degraded_path_is_paced_to_the_frame_interval(monkeypatch):
    """The freeze loop must not free-run.

    ``recv()`` has no clock: on a healthy device the rate comes from
    ``cap.read()`` blocking, and a dead one returns instantly.
    """
    _stub_videoframe(monkeypatch)
    sleeps = _record_sleeps(monkeypatch)
    track, _ = _patched_track()
    track.fps = 30
    track.actual_fps = 30

    asyncio.run(track.recv())
    assert sleeps == [], "a successful read must not add latency"

    track.cap.read.return_value = (False, None)
    asyncio.run(track.recv())

    assert sleeps == [pytest.approx(1 / 30)]


def test_degraded_path_is_paced_before_the_no_cached_frame_return(monkeypatch):
    """Cold start against a dead source spins just as hard without pacing."""
    _stub_videoframe(monkeypatch)
    sleeps = _record_sleeps(monkeypatch)
    track, _ = _patched_track()
    track.cap.read.return_value = (False, None)

    assert asyncio.run(track.recv()) is None
    assert sleeps == [pytest.approx(1 / 30)]


@pytest.mark.parametrize(
    "actual_fps, fps", [(None, 0), (0, None), (None, None), (None, -1)]
)
def test_degraded_pacing_survives_unusable_fps(monkeypatch, actual_fps, fps):
    """A misconfigured rate must degrade to a default, not divide by zero.

    A negative rate is the quiet one: ``asyncio.sleep`` accepts it and returns
    immediately, so the loop free-runs with no error to point at.
    """
    _stub_videoframe(monkeypatch)
    sleeps = _record_sleeps(monkeypatch)
    track, _ = _patched_track()
    track.actual_fps = actual_fps
    track.fps = fps
    track.cap.read.return_value = (False, None)

    asyncio.run(track.recv())

    assert sleeps == [pytest.approx(1 / CV2VideoTrack._DEGRADED_FALLBACK_FPS)]


def test_degraded_pacing_follows_the_rate_time_base_declares(monkeypatch):
    """Freeze ticks must be emitted at the rate ``time_base`` is built from.

    ``pts`` is ``frame_count`` against ``1/(actual_fps or fps)``, and freeze
    ticks bump ``frame_count``. Pacing on the configured 30 while the camera
    negotiated 15 would run the media clock at 2x for the whole outage.
    """
    _stub_videoframe(monkeypatch)
    sleeps = _record_sleeps(monkeypatch)
    track, _ = _patched_track()
    track.actual_fps = 15.0
    track.fps = 30
    track.cap.read.return_value = (False, None)

    asyncio.run(track.recv())

    assert sleeps == [pytest.approx(1 / 15)]


def test_close_cancels_pending_reconnect(monkeypatch):
    """``close()`` must cancel an in-flight reconnect to avoid leaks on shutdown."""
    _stub_videoframe(monkeypatch)
    track, _ = _patched_track()

    started = asyncio.Event()

    async def fake_loop():
        started.set()
        await asyncio.Event().wait()

    async def scenario():
        loop = asyncio.get_running_loop()
        track._reconnect_task = loop.create_task(fake_loop())
        await started.wait()
        track.close()
        # Give the loop a tick to process the cancellation.
        await asyncio.sleep(0)
        return track._reconnect_task

    task = asyncio.run(scenario())
    assert task.cancelled() or task.done()
    track.cap.release.assert_called_once()
