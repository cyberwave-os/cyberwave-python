import asyncio

import numpy as np
import pytest


def test_cv2_track_sets_current_frame_after_recv() -> None:
    pytest.importorskip("cv2")
    from cyberwave.sensor.base_video import BaseVideoTrack
    from cyberwave.sensor.camera_cv2 import CV2VideoTrack

    class _FakeCap:
        def read(self):
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self):
            pass

        def isOpened(self):
            return True

        def get(self, _prop):
            return 0

        def set(self, _prop, _val):
            return True

    track = CV2VideoTrack.__new__(CV2VideoTrack)
    BaseVideoTrack.__init__(track)
    track.cap = _FakeCap()
    track.fps = 30
    track.actual_fps = 30
    track.time_reference = None
    track.frame_callback = None
    track.keyframe_interval = None
    track._frames_since_keyframe = 0
    track._consecutive_read_failures = 0
    track._last_good_frame_bgr = None
    track._logged_frame_info = False
    track._reconnect_task = None
    track._reconnect_lock = asyncio.Lock()
    track._warn_frame_callback = lambda *a, **k: None
    track._maybe_schedule_reconnect = lambda: None

    frame = asyncio.run(track.recv())
    assert frame is not None
    assert track._current_frame is not None
    assert track._current_frame.shape == (8, 8, 3)
