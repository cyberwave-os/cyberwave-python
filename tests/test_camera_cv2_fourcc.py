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

from cyberwave.sensor import camera_cv2  # noqa: E402
from cyberwave.sensor.camera_cv2 import CV2VideoTrack  # noqa: E402


@pytest.fixture(autouse=True)
def _skip_v4l2_self_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass the Linux V4L2 build-info self-test for every test.

    The self-test (``_assert_v4l2_backend_or_raise``) inspects the real
    ``cv2.getBuildInformation()`` on the host running the tests, which is
    irrelevant for the negotiation logic exercised here. The
    ``TestV4L2SelfTest`` class disables this fixture and drives the check
    directly via patched build info.
    """
    monkeypatch.setenv("CYBERWAVE_CAMERA_SKIP_V4L2_CHECK", "1")
    # Reset the module-level "logged backends once" flag so tests that
    # assert on log output don't depend on test ordering.
    camera_cv2._LOGGED_VIDEO_BACKENDS = False


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
        pinned_geometry: bool = False,
    ) -> None:
        self.opened = opened
        self.reported_fourcc_code = reported_fourcc_code
        self.width = width
        self.height = height
        self.fps = fps
        self._backend = backend
        # When True, ``set(CAP_PROP_FRAME_*)`` does not update the readback
        # values — simulating a backend that silently rejects the requested
        # geometry and leaves the capture at its native default. This is
        # the YUYV-1080p fallback scenario from CYB-1998.
        self._pinned_geometry = pinned_geometry
        self.sets: list[tuple[int, Any]] = []
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirror cv2 API
        return self.opened

    def set(self, prop_id: int, value: Any) -> bool:  # noqa: A003 - mirror cv2 API
        import cv2

        self.sets.append((prop_id, value))
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH and not self._pinned_geometry:
            self.width = int(value)
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT and not self._pinned_geometry:
            self.height = int(value)
        elif prop_id == cv2.CAP_PROP_FPS and not self._pinned_geometry:
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

    def test_linux_atomic_failure_sequential_geometry_match_trusts_mjpg(self):
        """When atomic V4L2 open fails (e.g. OpenCV built without V4L2),
        the sequential fallback path takes over. If the FOURCC readback is
        empty but the geometry matches the requested resolution, we trust
        the requested MJPG and do **not** destructively reopen — that's
        the regression behind CYB-1998. The reopen would drop the device
        to its kernel default (YUYV 1080p @ 5 fps over USB 2.0).
        """
        initial = _FakeCap(reported_fourcc_code=0)
        atomic_failed = _FakeCap(opened=False)
        # seq_cap reports empty FOURCC (FFmpeg V4L2 demuxer quirk) but
        # geometry matches the requested 1280x720.
        seq_cap = _FakeCap(reported_fourcc_code=0, width=1280, height=720)

        track, _calls = _make_track_with_captures(
            system="Linux", captures=[initial, atomic_failed, seq_cap]
        )

        assert atomic_failed.released is True
        assert track.cap is seq_cap
        assert seq_cap.released is False
        assert track._fourcc_fallback_reopen is False
        assert track._negotiated_fourcc_ascii == "MJPG"

    def test_linux_atomic_failure_sequential_geometry_mismatch_still_reopens(self):
        """Geometry mismatch IS a real failure signal — the device actually
        landed on a different resolution, which means FOURCC really did
        not stick. Reopen without override is still the right move.
        """
        initial = _FakeCap(reported_fourcc_code=0)
        atomic_failed = _FakeCap(opened=False)
        # Camera ignores cap.set() and stays at its native 1920x1080 YUYV.
        seq_cap = _FakeCap(
            reported_fourcc_code=0,
            width=1920,
            height=1080,
            pinned_geometry=True,
        )
        seq_retry = _FakeCap(
            reported_fourcc_code=_yuyv_code(),
            width=1920,
            height=1080,
            pinned_geometry=True,
        )

        track, _calls = _make_track_with_captures(
            system="Linux", captures=[initial, atomic_failed, seq_cap, seq_retry]
        )

        assert atomic_failed.released is True
        assert seq_cap.released is True
        assert track.cap is seq_retry
        assert track._fourcc_fallback_reopen is True
        assert track._negotiated_fourcc_ascii == "YUYV"

    def test_linux_informative_nonempty_mismatch_still_reopens_even_when_geometry_matches(
        self,
    ):
        """The geometry-trust is intentionally gated to an EMPTY readback.

        If the V4L2 backend returns an informative non-empty FOURCC (e.g.
        ``YUYV``) at the requested resolution, that readback is the
        ground truth — the camera honestly negotiated YUYV, and we MUST
        NOT silently claim MJPG in telemetry just because geometry
        matched. Reopen-without-override is still the right move so the
        downstream pipeline gets the real format.
        """
        initial = _FakeCap(reported_fourcc_code=0)
        atomic_failed = _FakeCap(opened=False)
        # Camera negotiated YUYV (non-empty informative readback) at the
        # requested 1280x720. Geometry matches but FOURCC does not.
        seq_cap = _FakeCap(reported_fourcc_code=_yuyv_code())
        seq_retry = _FakeCap(reported_fourcc_code=_yuyv_code())

        track, _calls = _make_track_with_captures(
            system="Linux", captures=[initial, atomic_failed, seq_cap, seq_retry]
        )

        assert seq_cap.released is True
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
    """RTSP / HTTP sources must never take the atomic V4L2 path, must not
    push WIDTH/HEIGHT/FPS to the cap (the FFMPEG backend silently drops
    those for URL inputs), and must not surface a geometry "mismatch"
    when the source serves its own native format."""

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

    def test_url_source_does_not_push_geometry_or_fps_to_cap(self):
        """``cap.set(WIDTH/HEIGHT/FPS)`` is silently ignored by the FFMPEG
        backend for URL inputs. We skip those calls entirely so the init
        log can report the source's actual dimensions instead of the
        values we wanted but couldn't set.
        """
        captured: list[_FakeCap] = []

        def _factory(*_args: Any, **_kwargs: Any) -> _FakeCap:
            cap = _FakeCap(reported_fourcc_code=0, backend="FFMPEG")
            captured.append(cap)
            return cap

        with (
            patch("cyberwave.sensor.camera_cv2.platform.system", return_value="Linux"),
            patch("cyberwave.sensor.camera_cv2.cv2.VideoCapture", side_effect=_factory),
        ):
            CV2VideoTrack(
                camera_id="http://camera.example/snapshot.mjpg",
                fps=30,
                resolution=(1280, 720),
            )

        import cv2

        assert captured, "expected at least one cv2.VideoCapture call"
        forbidden_props = {
            cv2.CAP_PROP_FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT,
            cv2.CAP_PROP_FPS,
            cv2.CAP_PROP_FOURCC,
        }
        for cap in captured:
            pushed_props = {prop for prop, _value in cap.sets}
            offending = pushed_props & forbidden_props
            assert not offending, (
                f"URL source must not push {offending!r} to cap; got "
                f"sets={cap.sets!r}"
            )

    def test_url_source_actual_minus_requested_does_not_warn(
        self, caplog: pytest.LogCaptureFixture
    ):
        """When the source serves a different resolution from what the
        caller passed (the rule, not the exception, for URL inputs), we
        must NOT emit the legacy "Camera resolution mismatch" warning —
        the caller didn't fail, the SDK did the only correct thing
        (accept the source's geometry).
        """
        cap = _FakeCap(
            reported_fourcc_code=0,
            backend="FFMPEG",
            width=1920,
            height=1080,
            pinned_geometry=True,
        )

        with (
            patch("cyberwave.sensor.camera_cv2.platform.system", return_value="Linux"),
            patch(
                "cyberwave.sensor.camera_cv2.cv2.VideoCapture",
                side_effect=lambda *args, **kwargs: cap,
            ),
            caplog.at_level("INFO", logger="cyberwave.sensor.camera_cv2"),
        ):
            track = CV2VideoTrack(
                camera_id="http://camera.example/snapshot.mjpg",
                fps=30,
                resolution=(1280, 720),
            )

        assert track.actual_width == 1920
        assert track.actual_height == 1080
        assert "resolution mismatch" not in caplog.text.lower()
        assert "URL source" in caplog.text
        assert "source-served=1920x1080" in caplog.text

    def test_url_source_strict_geometry_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """``CYBERWAVE_CAMERA_STRICT_GEOMETRY=1`` exists for *local* edge
        images where the SDK genuinely controls the camera and a silent
        downgrade is a real bug (CYB-1998). For URL sources the SDK has
        no control over the upstream format, so the strict gate must
        be skipped — otherwise enabling it on an edge node that happens
        to consume an HTTP camera stream would crash on first init.
        """
        monkeypatch.setenv("CYBERWAVE_CAMERA_STRICT_GEOMETRY", "1")
        cap = _FakeCap(
            reported_fourcc_code=0,
            backend="FFMPEG",
            width=1920,
            height=1080,
            pinned_geometry=True,
        )

        with (
            patch("cyberwave.sensor.camera_cv2.platform.system", return_value="Linux"),
            patch(
                "cyberwave.sensor.camera_cv2.cv2.VideoCapture",
                side_effect=lambda *args, **kwargs: cap,
            ),
        ):
            # Must not raise.
            track = CV2VideoTrack(
                camera_id="http://camera.example/snapshot.mjpg",
                fps=30,
                resolution=(1280, 720),
            )

        assert track.actual_width == 1920
        assert track.actual_height == 1080


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
        """When geometry mismatch forces a destructive reopen, the
        stream attributes must surface the fact via
        ``fourcc_fallback_open_cv_default`` so operators can correlate a
        downgraded stream with this fallback path.
        """
        initial = _FakeCap(reported_fourcc_code=0)
        atomic_failed = _FakeCap(opened=False)
        # ``pinned_geometry=True`` simulates a camera that refuses
        # ``cap.set(WIDTH/HEIGHT)`` and stays at its native 1920x1080 —
        # the YUYV-1080p-5fps fallback in CYB-1998. Geometry mismatch
        # then triggers the reopen-without-FOURCC path.
        seq_cap = _FakeCap(
            reported_fourcc_code=0,
            width=1920,
            height=1080,
            pinned_geometry=True,
        )
        seq_retry = _FakeCap(
            reported_fourcc_code=_yuyv_code(),
            width=1920,
            height=1080,
            pinned_geometry=True,
        )

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
        """When the sequential path takes over and geometry matches (so
        no destructive reopen happens), the single sequential capture
        still receives BUFFERSIZE=2.
        """
        initial = _FakeCap(reported_fourcc_code=0)
        atomic_failed = _FakeCap(opened=False)
        seq_cap = _FakeCap(reported_fourcc_code=0, width=1280, height=720)

        _track, _calls = _make_track_with_captures(
            system="Linux",
            captures=[initial, atomic_failed, seq_cap],
        )

        assert self._buffersize_set_values(seq_cap) == [2]

    def test_linux_sequential_fallback_with_reopen_sets_buffersize_on_both(self):
        """When the sequential path reopens without FOURCC (geometry
        mismatched, real failure), both captures must request
        BUFFERSIZE=2 to avoid the uvcvideo 30fps -> 15fps halving.
        """
        initial = _FakeCap(reported_fourcc_code=0)
        atomic_failed = _FakeCap(opened=False)
        seq_cap = _FakeCap(
            reported_fourcc_code=0,
            width=1920,
            height=1080,
            pinned_geometry=True,
        )
        seq_retry = _FakeCap(
            reported_fourcc_code=_yuyv_code(),
            width=1920,
            height=1080,
            pinned_geometry=True,
        )

        _track, _calls = _make_track_with_captures(
            system="Linux",
            captures=[initial, atomic_failed, seq_cap, seq_retry],
        )

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


class TestV4L2SelfTest:
    """``CV2VideoTrack.__init__`` must refuse to start on Linux when the
    installed OpenCV lacks the V4L2 backend.

    Without V4L2 the SDK falls through to FFmpeg's libavformat V4L2
    demuxer, ``cap.set(CAP_PROP_FOURCC)`` becomes a no-op, and the camera
    silently downgrades to its kernel default (often YUYV 1920x1080
    @ 5 fps on USB 2.0). The check exists so that the first sign of a
    bad image is a `RuntimeError` at construction, not 5 fps in
    production. See CYB-1998.
    """

    _BUILD_INFO_WITHOUT_V4L2 = (
        "General configuration for OpenCV 4.13.0\n"
        "  Video I/O:\n"
        "    FFMPEG:                      YES\n"
        "    GStreamer:                   NO\n"
    )

    _BUILD_INFO_WITH_V4L2 = (
        "General configuration for OpenCV 4.13.0\n"
        "  Video I/O:\n"
        "    V4L/V4L2:                    YES\n"
        "    FFMPEG:                      YES\n"
        "    GStreamer:                   NO\n"
    )

    # Debian bookworm's python3-opencv 4.6.0 emits the row in lowercase,
    # which historically broke a case-sensitive build-time check (the
    # original CYB-1998 Dockerfile assertion was rejected by CI for this
    # reason). The regex MUST be case-insensitive.
    _BUILD_INFO_WITH_V4L2_DEBIAN_LOWERCASE = (
        "General configuration for OpenCV 4.6.0\n"
        "  Video I/O:\n"
        "    DC1394:                      YES (2.2.6)\n"
        "    FFMPEG:                      YES\n"
        "    GStreamer:                   YES (1.22.0)\n"
        "    PvAPI:                       NO\n"
        "    v4l/v4l2:                    YES (linux/videodev2.h)\n"
        "    gPhoto2:                     YES\n"
    )

    def test_linux_raises_when_v4l2_missing_from_build_info(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The actual regression we ship the check against."""
        monkeypatch.delenv("CYBERWAVE_CAMERA_SKIP_V4L2_CHECK", raising=False)
        with (
            patch(
                "cyberwave.sensor.camera_cv2.platform.system", return_value="Linux"
            ),
            patch(
                "cyberwave.sensor.camera_cv2.cv2.getBuildInformation",
                return_value=self._BUILD_INFO_WITHOUT_V4L2,
            ),
        ):
            with pytest.raises(RuntimeError) as excinfo:
                CV2VideoTrack(camera_id=0, fps=30, resolution=(1280, 720))
            msg = str(excinfo.value)
            assert "V4L2" in msg
            assert "CYBERWAVE_CAMERA_SKIP_V4L2_CHECK" in msg

    def test_linux_passes_when_v4l2_present_in_build_info(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The check must not false-positive on a correctly built OpenCV."""
        monkeypatch.delenv("CYBERWAVE_CAMERA_SKIP_V4L2_CHECK", raising=False)
        with (
            patch(
                "cyberwave.sensor.camera_cv2.platform.system", return_value="Linux"
            ),
            patch(
                "cyberwave.sensor.camera_cv2.cv2.getBuildInformation",
                return_value=self._BUILD_INFO_WITH_V4L2,
            ),
        ):
            from cyberwave.sensor.camera_cv2 import (
                _assert_v4l2_backend_or_raise,
            )

            _assert_v4l2_backend_or_raise()  # must not raise

    def test_linux_passes_on_debian_lowercase_v4l2_row(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Debian bookworm's ``python3-opencv`` 4.6.0 emits the row
        as lowercase ``v4l/v4l2:                    YES
        (linux/videodev2.h)``. The case-sensitive regex shipped in the
        first CYB-1998 draft rejected it and broke the camera-driver
        image build in CI. The check must be case-insensitive.
        """
        monkeypatch.delenv("CYBERWAVE_CAMERA_SKIP_V4L2_CHECK", raising=False)
        with (
            patch(
                "cyberwave.sensor.camera_cv2.platform.system", return_value="Linux"
            ),
            patch(
                "cyberwave.sensor.camera_cv2.cv2.getBuildInformation",
                return_value=self._BUILD_INFO_WITH_V4L2_DEBIAN_LOWERCASE,
            ),
        ):
            from cyberwave.sensor.camera_cv2 import (
                _assert_v4l2_backend_or_raise,
            )

            _assert_v4l2_backend_or_raise()  # must not raise

    def test_skip_env_var_bypasses_check(self, monkeypatch: pytest.MonkeyPatch):
        """``CYBERWAVE_CAMERA_SKIP_V4L2_CHECK=1`` is the documented escape
        hatch and must short-circuit before ``cv2.getBuildInformation``
        is even queried.
        """
        monkeypatch.setenv("CYBERWAVE_CAMERA_SKIP_V4L2_CHECK", "1")
        with (
            patch(
                "cyberwave.sensor.camera_cv2.platform.system", return_value="Linux"
            ),
            patch(
                "cyberwave.sensor.camera_cv2.cv2.getBuildInformation"
            ) as mock_info,
        ):
            from cyberwave.sensor.camera_cv2 import _assert_v4l2_backend_or_raise

            _assert_v4l2_backend_or_raise()
            mock_info.assert_not_called()

    def test_non_linux_platforms_skip_check(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """macOS and Windows backends are unaffected — they don't use V4L2
        and don't suffer the readback bug. The check must short-circuit
        on those platforms regardless of the env var.
        """
        monkeypatch.delenv("CYBERWAVE_CAMERA_SKIP_V4L2_CHECK", raising=False)
        for system in ("Darwin", "Windows"):
            with (
                patch(
                    "cyberwave.sensor.camera_cv2.platform.system",
                    return_value=system,
                ),
                patch(
                    "cyberwave.sensor.camera_cv2.cv2.getBuildInformation",
                    return_value=self._BUILD_INFO_WITHOUT_V4L2,
                ),
            ):
                from cyberwave.sensor.camera_cv2 import (
                    _assert_v4l2_backend_or_raise,
                )

                _assert_v4l2_backend_or_raise()  # must not raise

    def test_url_stream_bypasses_v4l2_check(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """RTSP / HTTP cameras don't open through V4L2 — they go through
        FFmpeg or GStreamer — so the V4L2 build flag is irrelevant for
        them. ``__init__`` must not raise on URL sources even when V4L2
        is absent from the build.
        """
        monkeypatch.delenv("CYBERWAVE_CAMERA_SKIP_V4L2_CHECK", raising=False)

        with (
            patch(
                "cyberwave.sensor.camera_cv2.platform.system", return_value="Linux"
            ),
            patch(
                "cyberwave.sensor.camera_cv2.cv2.getBuildInformation",
                return_value=self._BUILD_INFO_WITHOUT_V4L2,
            ),
            patch(
                "cyberwave.sensor.camera_cv2.cv2.VideoCapture",
                side_effect=lambda *_a, **_kw: _FakeCap(
                    reported_fourcc_code=0, backend="FFMPEG"
                ),
            ),
        ):
            track = CV2VideoTrack(
                camera_id="rtsp://camera.example/stream",
                fps=30,
                resolution=(1280, 720),
            )

        assert track.cap is not None


class TestDarwinFlowIsUnchangedByCyb1998:
    """The user-supplied ``fourcc=`` empty-readback guard is preserved
    cross-platform.

    CYB-1998 originally proposed gating this guard to Linux only on the
    grounds that AVFoundation reports FOURCC correctly. We deliberately
    DID NOT make that change because it would alter long-standing macOS
    behavior — when AVFoundation does report empty for a successful
    MJPG negotiation (which has been observed on some UVC cameras), the
    OLD code kept the capture and trusted MJPG; reopening without
    override would unnecessarily drop the user back to a default
    format. The new geometry-based trust is purely additive on Linux
    and does not fire on Darwin.
    """

    def test_macos_user_fourcc_empty_readback_preserves_old_trust_behaviour(self):
        """On Darwin, the SDK keeps the original behavior: explicit
        ``fourcc="MJPG"`` plus an empty readback is treated as a
        successful negotiation (trust the requested FOURCC, no
        destructive reopen). This must not change with CYB-1998.
        """
        # Only one capture should be consumed — if the SDK reopened we
        # would run out of fakes and the iterator would StopIteration.
        first = _FakeCap(reported_fourcc_code=0)

        track, _calls = _make_track_with_captures(
            system="Darwin",
            captures=[first],
            fourcc="MJPG",
        )

        assert first.released is False
        assert track.cap is first
        assert track._fourcc_fallback_reopen is False
        # Telemetry: explicit user_fourcc fallback preserves OLD behavior
        # of advertising the caller-requested tag.
        assert track._negotiated_fourcc_ascii == "MJPG"

    def test_macos_auto_mjpg_path_unchanged(self):
        """The auto-MJPG default on Darwin must keep its original
        semantics: try MJPG, accept whatever AVFoundation actually
        negotiates. The new Linux-only geometry trust must not fire here.
        """
        first = _FakeCap(reported_fourcc_code=_mjpg_code())

        track, _calls = _make_track_with_captures(system="Darwin", captures=[first])

        assert first.released is False
        assert track.cap is first
        assert track._fourcc_auto_mjpg is True
        assert track._fourcc_fallback_reopen is False
        assert track._negotiated_fourcc_ascii == "MJPG"

    def test_macos_fourcc_mismatch_with_nonempty_readback_still_reopens(self):
        """Genuine non-empty mismatch on Darwin (AVFoundation reported
        YUYV when we asked for MJPG) still triggers the reopen-without-
        override path. That behavior is unchanged.
        """
        first = _FakeCap(reported_fourcc_code=_yuyv_code())
        retry = _FakeCap(reported_fourcc_code=_yuyv_code())

        track, _calls = _make_track_with_captures(
            system="Darwin", captures=[first, retry]
        )

        assert first.released is True
        assert track.cap is retry
        assert track._fourcc_fallback_reopen is True
        assert track._negotiated_fourcc_ascii == "YUYV"


class TestStrictGeometryEscalation:
    """``CYBERWAVE_CAMERA_STRICT_GEOMETRY=1`` promotes a silent resolution
    mismatch to a hard ``RuntimeError``.

    Default behaviour stays a WARNING because some cameras legitimately
    round to the nearest supported mode. The flag exists for edge images
    where we control the camera/config and want a 56x bandwidth blowup
    to fail loudly instead of shipping a "working" stream.
    """

    def test_default_off_keeps_warning_behaviour(self):
        first = _FakeCap(
            reported_fourcc_code=_mjpg_code(),
            width=1920,
            height=1080,
            pinned_geometry=True,
        )

        track, _calls = _make_track_with_captures(
            system="Darwin", captures=[first], resolution=(1280, 720)
        )

        # No exception; track exposes the actual (mismatched) geometry.
        assert track.actual_width == 1920
        assert track.actual_height == 1080

    def test_env_var_promotes_mismatch_to_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CYBERWAVE_CAMERA_STRICT_GEOMETRY", "1")
        first = _FakeCap(
            reported_fourcc_code=_mjpg_code(),
            width=1920,
            height=1080,
            pinned_geometry=True,
        )

        with pytest.raises(RuntimeError) as excinfo:
            _make_track_with_captures(
                system="Darwin", captures=[first], resolution=(1280, 720)
            )

        assert "resolution mismatch" in str(excinfo.value).lower()
        assert "CYBERWAVE_CAMERA_STRICT_GEOMETRY" in str(excinfo.value)
