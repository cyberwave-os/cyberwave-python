"""Tests for improved timestamp capture in camera tracks."""

import time

import pytest

from cyberwave.utils import TimeReference


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
