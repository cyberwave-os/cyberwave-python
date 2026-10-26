"""Tests for improved timestamp capture in camera tracks."""

import sys
import time

import pytest

from cyberwave.utils import TimeReference, time_reference


class TestTimestampCapture:
    """Test TimeReference and camera track timestamp behavior."""

    def test_time_reference_read_returns_cached_after_update(self):
        """read() returns last values from update(); does not sample time by itself."""
        ref = TimeReference()

        t1, tm1 = ref.update()

        time.sleep(0.01)

        t2, tm2 = ref.read()
        assert t2 == t1
        assert tm2 == tm1

        t3, tm3 = ref.update()
        assert t3 > t1
        assert tm3 > tm1

        t4, tm4 = ref.read()
        assert t4 == t3
        assert tm4 == tm3

    @pytest.mark.anyio(backends=["asyncio"])
    async def test_virtual_track_metadata_storage(self):
        """Test that virtual track stores per-frame metadata for the sync extension."""
        pytest.importorskip(
            "av",
            reason="av not installed (install with extras: camera)",
        )
        pytest.importorskip(
            "aiortc",
            reason="aiortc not installed (install with extras: camera)",
        )
        import numpy as np
        from cyberwave.sensor.camera_virtual import VirtualVideoTrack

        def get_frame():
            return np.zeros((480, 640, 3), dtype=np.uint8)

        track = VirtualVideoTrack(
            get_frame=get_frame,
            width=640,
            height=480,
            fps=30,
        )

        # Produce a frame
        frame = await track.recv()

        # Verify metadata was stored
        assert hasattr(track, "_current_frame_index")
        assert hasattr(track, "_current_pts")
        assert hasattr(track, "_current_time_base_num")
        assert hasattr(track, "_current_time_base_den")
        assert hasattr(track, "_current_capture_wall_time")
        assert hasattr(track, "_current_capture_monotonic")

        # Verify values make sense
        assert track._current_frame_index == 0
        assert track._current_pts == 0
        assert track._current_time_base_num == 1
        assert track._current_time_base_den == 30
        assert track._current_capture_wall_time > 0
        assert track._current_capture_monotonic > 0


class TestAcquisitionClock:
    """Which monotonic clock acquisition stamps are read from."""

    # Stated independently of the SDK's own selection, so this pins a fact about
    # the platform rather than restating the code under test.
    SUSPEND_INCLUSIVE = {"linux": "CLOCK_BOOTTIME", "darwin": "CLOCK_MONOTONIC"}

    def test_stamps_from_the_suspend_inclusive_clock(self):
        """The monotonic half tracks a clock that keeps running across suspend."""
        expected = self.SUSPEND_INCLUSIVE.get(sys.platform)
        if expected is None:
            pytest.skip(f"no suspend-inclusive clock documented for {sys.platform}")

        _, monotonic = TimeReference().update()

        # Off by the host's whole accumulated suspend when stamped from an
        # awake-only clock.
        assert monotonic == pytest.approx(
            time.clock_gettime(getattr(time, expected)), abs=0.5
        )
        assert time_reference.MONOTONIC_CLOCK_NAME == expected
        assert time_reference.MONOTONIC_CLOCK_IS_SUSPEND_INCLUSIVE

    def test_helper_is_reachable_from_the_package(self):
        """Producers outside the SDK import it from ``cyberwave.utils``.

        ``cyberwave-sim`` stamps its rendered frames with this helper so a simulated
        camera and a physical one put one domain on the wire; that import has to
        resolve against the package, not a private module path.
        """
        from cyberwave import utils

        assert utils.monotonic_seconds is time_reference.monotonic_seconds
        assert utils.MONOTONIC_CLOCK_NAME == time_reference.MONOTONIC_CLOCK_NAME
        assert (
            utils.MONOTONIC_CLOCK_IS_SUSPEND_INCLUSIVE
            is time_reference.MONOTONIC_CLOCK_IS_SUSPEND_INCLUSIVE
        )
