"""Integration tests for the sync extension."""

import pytest


def test_sdk_imports_without_cyberwave_video_sync():
    """SDK's base_video module imports without cyberwave_video_sync installed."""
    from cyberwave.sensor.base_video import BaseVideoTrack
    assert hasattr(BaseVideoTrack, "_store_frame_metadata_for_sync")


def test_sync_extension_install_when_present():
    """If cyberwave_video_sync is installed, install() works without error."""
    try:
        import cyberwave_video_sync
        cyberwave_video_sync.install()
        cyberwave_video_sync.install()  # Should be idempotent
    except ImportError:
        pytest.skip("cyberwave_video_sync not installed")


def test_sync_extension_emit_when_present():
    """If cyberwave_video_sync is installed, emit() works correctly."""
    try:
        import cyberwave_video_sync
    except ImportError:
        pytest.skip("cyberwave_video_sync not installed")

    class MockTrack:
        id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        _sync_enabled = True
        _current_frame_index = 0
        _current_pts = 0
        _current_time_base_num = 1
        _current_time_base_den = 30
        _current_capture_wall_time = 1680000000.0
        _current_capture_monotonic = 123456.789

    track = MockTrack()
    input_packets = [b"\x07test_sps", b"\x08test_pps", b"\x65test_idr"]
    result = cyberwave_video_sync.emit(track, input_packets, force_keyframe=True)

    assert isinstance(result, list)
    assert len(result) >= len(input_packets)
