"""Regression tests for ``CV2VideoTrack`` FOURCC negotiation.

These tests pin the behaviour introduced to work around two OpenCV V4L2
backend quirks that caused MJPG negotiation to spuriously fail on uvcvideo
devices:

* After sequential ``cap.set(CAP_PROP_FOURCC, ...)`` the readback often
  returns ``0``/empty even though the format was applied, which made the
  SDK treat a successful MJPG set as a failure and drop to YUYV @ 10fps.
* Sequential ``cap.set(WIDTH/HEIGHT)`` at HD resolutions can leave the
  capture unresponsive.

The fix uses the ``cv2.VideoCapture(src, CAP_V4L2, [params...])`` atomic
constructor on Linux local devices and keeps the legacy sequential path
(with the FOURCC-did-not-stick reopen fallback) for every other code
path.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("cv2", reason="OpenCV not installed")
pytest.importorskip("av", reason="pyav not installed")

from cyberwave.sensor.camera_cv2 import CV2VideoTrack  # noqa: E402


class _FakeCap:
    """Minimal ``cv2.VideoCapture`` stand-in used by the tests.

    We mimic only the subset of the API that the negotiation code path
    exercises: ``isOpened``, ``set``, ``get`` (with a configurable FOURCC
    readback), ``release``, and ``getBackendName``.
    """

    def __init__(
        self,
        *,
        opened: bool = True,
        reported_fourcc_code: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        backend: str = "V4L2",
    ) -> None:
        self.opened = opened
        self.reported_fourcc_code = reported_fourcc_code
        self.width = width
        self.height = height
        self.fps = fps
        self._backend = backend
        self.sets: list[tuple[int, Any]] = []
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirror cv2 API
        return self.opened

    def set(self, prop_id: int, value: Any) -> bool:  # noqa: A003 - mirror cv2 API
        import cv2

        self.sets.append((prop_id, value))
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            self.width = int(value)
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            self.height = int(value)
        elif prop_id == cv2.CAP_PROP_FPS:
            self.fps = float(value)
        elif prop_id == cv2.CAP_PROP_FOURCC:
            # By default we do NOT update the readback value: that's the
            # exact OpenCV/V4L2 bug we are working around on the sequential
            # path.  Individual tests override ``reported_fourcc_code`` to
            # simulate a backend that does report the negotiated code.
            pass
        return True

    def get(self, prop_id: int) -> float:
        import cv2

        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        if prop_id == cv2.CAP_PROP_FOURCC:
            return float(self.reported_fourcc_code)
        return 0.0

    def getBackendName(self) -> str:  # noqa: N802 - mirror cv2 API
        return self._backend

    def release(self) -> None:
        self.released = True
        self.opened = False


def _mjpg_code() -> int:
    import cv2

    return int(cv2.VideoWriter_fourcc(*"MJPG"))


def _yuyv_code() -> int:
    import cv2

    return int(cv2.VideoWriter_fourcc(*"YUYV"))


def _make_track_with_captures(
    *,
    system: str,
    captures: list[_FakeCap],
    fourcc: str | None = None,
    camera_id: int | str = 0,
    resolution: tuple[int, int] = (1280, 720),
    fps: int = 30,
) -> tuple[CV2VideoTrack, list[tuple[tuple, dict]]]:
    """Instantiate ``CV2VideoTrack`` with ``cv2.VideoCapture`` mocked out.

    Returns the constructed track and the list of ``(args, kwargs)`` tuples
    captured by each ``cv2.VideoCapture`` call so tests can assert which
    constructor overload was used.
    """
    calls: list[tuple[tuple, dict]] = []
    iterator = iter(captures)

    def _factory(*args: Any, **kwargs: Any) -> _FakeCap:
        calls.append((args, kwargs))
        try:
            return next(iterator)
        except StopIteration:  # pragma: no cover - defensive
            raise AssertionError(
                "cv2.VideoCapture called more times than the test prepared fake captures for"
            ) from None

    with (
        patch("cyberwave.sensor.camera_cv2.platform.system", return_value=system),
        patch("cyberwave.sensor.camera_cv2.cv2.VideoCapture", side_effect=_factory),
    ):
        track = CV2VideoTrack(
            camera_id=camera_id,
            fps=fps,
            resolution=resolution,
            fourcc=fourcc,
        )
    return track, calls


class TestAtomicV4L2Open:
    """Linux local cameras should use the atomic open-time params path."""

    def test_linux_uses_atomic_v4l2_constructor_for_default_mjpg(self):
        # First call: initial _open_capture() — returns a plain fake capture
        # that negotiation will release before reopening atomically.
        initial = _FakeCap(reported_fourcc_code=0)
        # Second call: atomic open succeeds and reports MJPG back.
        atomic = _FakeCap(reported_fourcc_code=_mjpg_code())

        track, calls = _make_track_with_captures(
            system="Linux", captures=[initial, atomic]
        )

        assert initial.released is True
        assert track.cap is atomic
        assert track._fourcc_auto_mjpg is True
        assert track._fourcc_attempted == "MJPG"
        assert track._negotiated_fourcc_ascii == "MJPG"
        # No sequential fallback should have been triggered.
        assert track._fourcc_fallback_reopen is False

        # Second VideoCapture invocation is the atomic one: (camera_id,
        # CAP_V4L2, [params...]).
        import cv2

        atomic_args, atomic_kwargs = calls[1]
        assert atomic_args[0] == 0
        assert atomic_args[1] == cv2.CAP_V4L2
        assert isinstance(atomic_args[2], list)
        assert cv2.CAP_PROP_FOURCC in atomic_args[2]
        assert cv2.CAP_PROP_FRAME_WIDTH in atomic_args[2]
        assert cv2.CAP_PROP_FRAME_HEIGHT in atomic_args[2]
        assert cv2.CAP_PROP_FPS in atomic_args[2]
        assert atomic_kwargs == {}

    def test_linux_atomic_open_tolerates_empty_fourcc_readback(self):
        """Empty FOURCC readback after atomic open must NOT trigger a fallback.

        This is the core regression: uvcvideo reports ``0`` from
        ``CAP_PROP_FOURCC`` even when MJPG was applied.  The atomic path
        trusts the open-time params and should not reopen without FOURCC.
        Because the ``CAP_V4L2`` constructor either accepts the requested
        FOURCC or fails the open, we also trust ``effective_try`` for the
        WebRTC offer ``fourcc`` field when the readback is empty.
        """
        initial = _FakeCap(reported_fourcc_code=0)
        atomic = _FakeCap(reported_fourcc_code=0)  # buggy empty readback

        track, _calls = _make_track_with_captures(
            system="Linux", captures=[initial, atomic]
        )

        assert track.cap is atomic
        assert track._fourcc_fallback_reopen is False
        # Trust the requested FOURCC on atomic success even when the
        # backend readback is empty — otherwise downstream consumers see
        # no ``fourcc`` in the stream attributes despite MJPG being live.
        assert track._negotiated_fourcc_ascii == "MJPG"

    def test_linux_user_supplied_fourcc_is_honoured_and_not_auto_flagged(self):
        """Explicit ``fourcc="YUYV"`` must be passed through atomically and
        must NOT set ``_fourcc_auto_mjpg`` — that flag only signals the
        SDK-side default.  The WebRTC offer must also carry the user's
        choice via ``fourcc_requested``.
        """
        initial = _FakeCap(reported_fourcc_code=0)
        atomic = _FakeCap(reported_fourcc_code=_yuyv_code())

        track, calls = _make_track_with_captures(
            system="Linux",
            captures=[initial, atomic],
            fourcc="YUYV",
        )

        assert track.cap is atomic
        assert track._fourcc_auto_mjpg is False
        assert track._fourcc_attempted == "YUYV"
        assert track._negotiated_fourcc_ascii == "YUYV"
        assert track._fourcc_fallback_reopen is False

        import cv2

        atomic_args, _ = calls[1]
        assert atomic_args[1] == cv2.CAP_V4L2
        params = atomic_args[2]
        fourcc_value_index = params.index(cv2.CAP_PROP_FOURCC) + 1
        assert params[fourcc_value_index] == cv2.VideoWriter_fourcc(*"YUYV")

        attrs = track.get_stream_attributes()
        assert attrs["fourcc"] == "YUYV"
        assert attrs["fourcc_requested"] == "YUYV"
        assert "fourcc_auto_mjpg" not in attrs

    def test_linux_atomic_open_failure_falls_back_to_sequential(self):
        """If the atomic open returns an unopened capture we must recover."""
        initial = _FakeCap(reported_fourcc_code=0)
        atomic_failed = _FakeCap(opened=False)
        # Sequential fallback: first the reopened default capture, then the
        # "FOURCC did not stick" reopen without override.
        seq_cap = _FakeCap(reported_fourcc_code=0)
        seq_retry = _FakeCap(reported_fourcc_code=_yuyv_code())

        track, _calls = _make_track_with_captures(
            system="Linux", captures=[initial, atomic_failed, seq_cap, seq_retry]
        )

        assert atomic_failed.released is True
        assert track.cap is seq_retry
        assert track._fourcc_fallback_reopen is True
        assert track._negotiated_fourcc_ascii == "YUYV"

    def test_linux_user_mjpg_atomic_fails_empty_readback_skips_destructive_reopen(self):
        """Explicit ``fourcc=`` in config: empty CAP_PROP_FOURCC must not drop MJPG.

        Matches uvcvideo in Docker: atomic open can fail, sequential set applies
        MJPG but readback is empty; reopening without FOURCC regresses to 1080p@5
        YUYV — we keep the first sequential capture and trust the requested tag.
        """
        initial = _FakeCap(reported_fourcc_code=0)
        atomic_failed = _FakeCap(opened=False)
        seq_cap = _FakeCap(reported_fourcc_code=0)  # empty FOURCC readback

        track, _calls = _make_track_with_captures(
            system="Linux",
            captures=[initial, atomic_failed, seq_cap],
            fourcc="MJPG",
        )

        assert track.cap is seq_cap
        assert seq_cap.released is False
        assert track._fourcc_auto_mjpg is False
        assert track._fourcc_fallback_reopen is False
        assert track._negotiated_fourcc_ascii == "MJPG"


class TestSequentialPath:
    """Non-Linux platforms and URL streams keep the legacy sequential path."""

    def test_macos_uses_sequential_cap_set_negotiation(self):
        initial = _FakeCap(reported_fourcc_code=_mjpg_code())

        track, calls = _make_track_with_captures(system="Darwin", captures=[initial])

        # Only the initial open — atomic reopen must NOT happen on macOS.
        assert len(calls) == 1
        assert track.cap is initial
        assert initial.released is False
        assert track._fourcc_auto_mjpg is True
        assert track._fourcc_fallback_reopen is False
        assert track._negotiated_fourcc_ascii == "MJPG"

    def test_macos_fourcc_mismatch_triggers_reopen_without_override(self):
        """If the backend reports a different FOURCC, we reopen without override."""
        # First capture reports YUYV even though we asked for MJPG.
        first = _FakeCap(reported_fourcc_code=_yuyv_code())
        # After the reopen we accept whatever the driver picks.
        retry = _FakeCap(reported_fourcc_code=_yuyv_code())

        track, calls = _make_track_with_captures(
            system="Darwin", captures=[first, retry]
        )

        assert first.released is True
        assert track.cap is retry
        assert track._fourcc_fallback_reopen is True
        assert track._negotiated_fourcc_ascii == "YUYV"
        # Neither capture was opened via CAP_V4L2 — both are sequential opens.
        for args, _kwargs in calls:
            if len(args) >= 2:
                import cv2

                assert args[1] != cv2.CAP_V4L2


class TestUrlStreams:
    """RTSP / HTTP sources must never take the atomic V4L2 path."""

    def test_rtsp_url_on_linux_stays_on_sequential_path(self):
        calls: list[tuple[tuple, dict]] = []

        def _factory(*args: Any, **kwargs: Any) -> _FakeCap:
            calls.append((args, kwargs))
            return _FakeCap(reported_fourcc_code=0, backend="FFMPEG")

        with (
            patch("cyberwave.sensor.camera_cv2.platform.system", return_value="Linux"),
            patch("cyberwave.sensor.camera_cv2.cv2.VideoCapture", side_effect=_factory),
        ):
            track = CV2VideoTrack(
                camera_id="rtsp://camera.example/stream",
                fps=30,
                resolution=(1280, 720),
            )

        import cv2

        for args, _kwargs in calls:
            if len(args) >= 2:
                assert args[1] != cv2.CAP_V4L2
        assert track._fourcc_auto_mjpg is False
        assert track._fourcc_attempted is None


class TestStreamAttributesRemainStable:
    """The WebRTC offer payload keys must keep their existing names/semantics.

    Downstream consumers (edge-core, workers) rely on the attribute dict
    returned by ``get_stream_attributes`` — notably the presence/absence
    of the ``fourcc_auto_mjpg`` and ``fourcc_fallback_open_cv_default``
    telemetry flags.  This test documents that contract.
    """

    def test_atomic_open_success_sets_auto_mjpg_only(self):
        initial = _FakeCap(reported_fourcc_code=0)
        atomic = _FakeCap(reported_fourcc_code=_mjpg_code())

        track, _calls = _make_track_with_captures(
            system="Linux", captures=[initial, atomic]
        )
        attrs = track.get_stream_attributes()

        assert attrs["fourcc"] == "MJPG"
        assert attrs["fourcc_auto_mjpg"] is True
        assert "fourcc_fallback_open_cv_default" not in attrs

    def test_atomic_failure_sets_fallback_flag(self):
        initial = _FakeCap(reported_fourcc_code=0)
        atomic_failed = _FakeCap(opened=False)
        seq_cap = _FakeCap(reported_fourcc_code=0)
        seq_retry = _FakeCap(reported_fourcc_code=_yuyv_code())

        track, _calls = _make_track_with_captures(
            system="Linux",
            captures=[initial, atomic_failed, seq_cap, seq_retry],
        )
        attrs = track.get_stream_attributes()

        assert attrs["fourcc_auto_mjpg"] is True
        assert attrs["fourcc_fallback_open_cv_default"] is True
        assert attrs["fourcc"] == "YUYV"


class TestBufferSize:
    """``CAP_PROP_BUFFERSIZE=1`` halves V4L2 framerate; we request 2 on Linux.

    Measured on uvcvideo + OpenCV 4.13: 30fps MJPG drops to exactly 15fps
    when ``BUFFERSIZE=1`` is set.  ``BUFFERSIZE>=2`` restores native
    throughput while keeping latency low (one decoded frame in hand, one
    captured in the kernel buffer).
    """

    @staticmethod
    def _buffersize_set_values(cap: _FakeCap) -> list[int]:
        import cv2

        return [value for prop, value in cap.sets if prop == cv2.CAP_PROP_BUFFERSIZE]

    def test_linux_atomic_open_sets_buffersize_to_two(self):
        initial = _FakeCap(reported_fourcc_code=0)
        atomic = _FakeCap(reported_fourcc_code=_mjpg_code())

        _track, _calls = _make_track_with_captures(
            system="Linux", captures=[initial, atomic]
        )

        assert self._buffersize_set_values(atomic) == [2]

    def test_linux_sequential_fallback_sets_buffersize_to_two(self):
        initial = _FakeCap(reported_fourcc_code=0)
        atomic_failed = _FakeCap(opened=False)
        seq_cap = _FakeCap(reported_fourcc_code=0)
        seq_retry = _FakeCap(reported_fourcc_code=_yuyv_code())

        _track, _calls = _make_track_with_captures(
            system="Linux",
            captures=[initial, atomic_failed, seq_cap, seq_retry],
        )

        # seq_cap saw the original FOURCC set + geometry; seq_retry saw the
        # reopen-without-override pass.  Both must request BUFFERSIZE=2.
        assert self._buffersize_set_values(seq_cap) == [2]
        assert self._buffersize_set_values(seq_retry) == [2]

    def test_non_linux_keeps_buffersize_one(self):
        initial = _FakeCap(reported_fourcc_code=_mjpg_code())

        _track, _calls = _make_track_with_captures(system="Darwin", captures=[initial])

        assert self._buffersize_set_values(initial) == [1]


class TestAtomicOpenErrorHandling:
    """Atomic open must return ``None`` (not raise) on builds that lack the
    ``CAP_V4L2`` symbol or the params overload, so the caller can cleanly
    fall back to the legacy sequential path.
    """

    def _stub_track(self) -> CV2VideoTrack:
        track = CV2VideoTrack.__new__(CV2VideoTrack)
        track.camera_id = 0
        track.requested_width = 1280
        track.requested_height = 720
        track.fps = 30
        return track

    def test_returns_none_on_typeerror_old_opencv(self):
        """OpenCV < 4.5 lacks the ``(src, backend, params)`` overload."""
        track = self._stub_track()

        def _raise(*_args: Any, **_kwargs: Any) -> Any:
            raise TypeError("VideoCapture() takes at most 2 arguments")

        with patch("cyberwave.sensor.camera_cv2.cv2.VideoCapture", side_effect=_raise):
            assert track._open_local_v4l2_atomic("MJPG") is None

    def test_returns_none_on_attributeerror_missing_cap_v4l2(self):
        """Builds without V4L2 support won't expose ``CAP_V4L2``."""
        track = self._stub_track()

        def _raise(*_args: Any, **_kwargs: Any) -> Any:
            raise AttributeError("module 'cv2' has no attribute 'CAP_V4L2'")

        with patch("cyberwave.sensor.camera_cv2.cv2.VideoCapture", side_effect=_raise):
            assert track._open_local_v4l2_atomic("MJPG") is None

    def test_end_to_end_falls_back_to_sequential_on_typeerror(self):
        """When the atomic overload is missing, the full track init path
        must still produce a working capture via the legacy sequential
        open."""
        calls: list[tuple[tuple, dict]] = []
        produced: list[_FakeCap] = []

        def _factory(*args: Any, **kwargs: Any) -> _FakeCap:
            calls.append((args, kwargs))
            # Single-arg opens (initial + fallback) succeed; the atomic
            # open (3 args) raises TypeError to simulate OpenCV < 4.5.
            if len(args) >= 2:
                raise TypeError("no params overload")
            cap = _FakeCap(reported_fourcc_code=_mjpg_code())
            produced.append(cap)
            return cap

        with (
            patch("cyberwave.sensor.camera_cv2.platform.system", return_value="Linux"),
            patch("cyberwave.sensor.camera_cv2.cv2.VideoCapture", side_effect=_factory),
        ):
            track = CV2VideoTrack(camera_id=0, fps=30, resolution=(1280, 720))

        # Exactly three VideoCapture invocations: initial open, failed
        # atomic open, sequential-fallback reopen.
        assert len(calls) == 3
        assert len(calls[1][0]) == 3  # atomic attempt
        assert len(calls[0][0]) == 1 and len(calls[2][0]) == 1  # sequential
        assert track.cap is produced[-1]
        assert track._negotiated_fourcc_ascii == "MJPG"
        assert track._fourcc_fallback_reopen is False


def test_should_use_atomic_v4l2_open_gates_on_platform_only():
    """The predicate is now purely a platform + source-type gate; the
    ``CAP_V4L2`` availability check lives inside ``_open_local_v4l2_atomic``
    so that failures surface as a clean ``None`` return rather than a
    predicate refusal."""
    track = CV2VideoTrack.__new__(CV2VideoTrack)
    track.camera_id = 0

    with patch("cyberwave.sensor.camera_cv2.platform.system", return_value="Linux"):
        assert track._should_use_atomic_v4l2_open() is True

    with patch("cyberwave.sensor.camera_cv2.platform.system", return_value="Darwin"):
        assert track._should_use_atomic_v4l2_open() is False

    with patch("cyberwave.sensor.camera_cv2.platform.system", return_value="Windows"):
        assert track._should_use_atomic_v4l2_open() is False

    track.camera_id = "rtsp://camera.example/stream"
    with patch("cyberwave.sensor.camera_cv2.platform.system", return_value="Linux"):
        assert track._should_use_atomic_v4l2_open() is False
