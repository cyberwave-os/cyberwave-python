"""Unit tests for VirtualCameraStreamer stream identity signaling."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

import pytest

pytest.importorskip(
    "aiortc", reason="aiortc not installed (install with extras: camera)"
)
sys.modules.setdefault(
    "yaml",
    types.SimpleNamespace(
        safe_load=lambda *args, **kwargs: {},
        dump=lambda *args, **kwargs: "",
    ),
)

# Optional camera dependencies must be guarded before importing their classes.
from cyberwave.sensor.base_video import DEFAULT_TURN_SERVERS  # noqa: E402
from cyberwave.sensor.camera_virtual import VirtualCameraStreamer, VirtualVideoTrack  # noqa: E402


def _make_mqtt_client(topic_prefix: str = "") -> MagicMock:
    client = MagicMock()
    client.topic_prefix = topic_prefix
    client.subscribe = MagicMock()
    client.publish = MagicMock()
    client.client_id = "test-client"
    return client


def _extract_on_answer(streamer: VirtualCameraStreamer):
    streamer._subscribe_to_answer()
    return streamer.client.subscribe.call_args_list[0][0][1]


class TestVirtualCameraStreamerIdentity:
    def _make_streamer(self) -> VirtualCameraStreamer:
        return VirtualCameraStreamer(
            client=_make_mqtt_client(),
            get_frame=lambda: None,
            twin_uuid="twin-123",
            camera_name="front_camera",
            stream_source="simulation",
        )

    def test_offer_includes_stream_identity(self):
        streamer = self._make_streamer()
        streamer.pc = SimpleNamespace(localDescription=SimpleNamespace(type="offer"))
        streamer.streamer = SimpleNamespace(
            id="track-1", get_stream_attributes=lambda: {}
        )

        streamer._send_offer("v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n")

        payload = streamer.client.publish.call_args[0][1]
        assert payload["sensor"] == "front_camera"
        assert payload["stream_source"] == "simulation"
        assert "stream_instance_id" not in payload

    def test_answer_with_wrong_stream_identity_is_rejected(self):
        streamer = self._make_streamer()
        on_answer = _extract_on_answer(streamer)

        on_answer(
            json.dumps(
                {
                    "type": "answer",
                    "target": "edge",
                    "sdp": "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n",
                    "sensor": "front_camera",
                    "stream_source": "simulation",
                    "stream_instance_id": "other-sim",
                }
            )
        )

        assert streamer._answer_received is False

    def test_matching_answer_with_stream_identity_is_accepted(self):
        streamer = self._make_streamer()
        on_answer = _extract_on_answer(streamer)

        payload = {
            "type": "answer",
            "target": "edge",
            "sdp": "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n",
            "sensor": "front_camera",
            "stream_source": "simulation",
        }
        on_answer(payload)

        assert streamer._answer_received is True
        assert streamer._answer_data == payload


def test_virtual_video_track_forces_periodic_keyframes():
    track = VirtualVideoTrack(
        get_frame=lambda: np.zeros((16, 16, 3), dtype=np.uint8),
        width=16,
        height=16,
        fps=1000,
        keyframe_interval=2,
    )

    first = asyncio.run(track.recv())
    second = asyncio.run(track.recv())
    third = asyncio.run(track.recv())
    fourth = asyncio.run(track.recv())

    assert first.key_frame == 1
    assert second.key_frame != 1
    assert third.key_frame != 1
    assert fourth.key_frame == 1
    assert first.format.name == "yuv420p"


def test_virtual_video_track_allows_configurable_output_format():
    track = VirtualVideoTrack(
        get_frame=lambda: np.zeros((16, 16, 3), dtype=np.uint8),
        width=16,
        height=16,
        fps=1000,
        output_format="rgb24",
    )

    frame = asyncio.run(track.recv())

    assert frame.format.name == "rgb24"


class TestVirtualCameraStreamerTurnServers:
    def test_none_uses_platform_default_turn_servers(self):
        streamer = VirtualCameraStreamer(
            client=_make_mqtt_client(),
            get_frame=lambda: None,
            twin_uuid="twin-123",
        )

        assert streamer.turn_servers == DEFAULT_TURN_SERVERS

    def test_empty_list_disables_turn_for_local_ice_only(self):
        streamer = VirtualCameraStreamer(
            client=_make_mqtt_client(),
            get_frame=lambda: None,
            twin_uuid="twin-123",
            turn_servers=[],
        )

        assert streamer.turn_servers == []

    def test_explicit_turn_servers_are_forwarded(self):
        custom_servers = [{"urls": "stun:turn.cyberwave.com:3478"}]
        streamer = VirtualCameraStreamer(
            client=_make_mqtt_client(),
            get_frame=lambda: None,
            twin_uuid="twin-123",
            turn_servers=custom_servers,
        )

        assert streamer.turn_servers == custom_servers


def test_virtual_camera_streamer_passes_output_format_to_track():
    streamer = VirtualCameraStreamer(
        client=_make_mqtt_client(),
        get_frame=lambda: np.zeros((16, 16, 3), dtype=np.uint8),
        width=16,
        height=16,
        fps=1000,
        output_format="rgb24",
    )

    track = streamer.initialize_track()
    frame = asyncio.run(track.recv())

    assert frame.format.name == "rgb24"


def test_virtual_sample_keeps_pixels_and_source_clock_together():
    from cyberwave.sensor.frame import CapturedVideoFrame

    pixels = np.full((8, 8, 3), 23, dtype=np.uint8)
    sample = CapturedVideoFrame(pixels, 1700000000.25, 12.5, 7)
    # Reusing a renderer buffer must not change an already captured sample.
    pixels[:] = 99
    reference = SimpleNamespace(read=lambda: (1900000000.0, 900.0))
    track = VirtualVideoTrack(
        lambda: sample, fps=1000, output_format="rgb24", time_reference=reference
    )

    first = asyncio.run(track.recv())
    second = asyncio.run(track.recv())

    assert np.all(first.to_ndarray(format="rgb24") == 23)
    assert np.all(second.to_ndarray(format="rgb24") == 23)
    assert track._current_capture_wall_time == 1700000000.25
    assert track._current_capture_monotonic == 12.5
    assert (first.pts, second.pts) == (0, 1)
    assert track.frame_count == 2  # sender identity remains distinct
    assert track.captured_frame_count == 1


def test_identical_pixels_from_new_acquisition_count_as_fresh():
    from cyberwave.sensor.frame import CapturedVideoFrame

    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    supplied = [CapturedVideoFrame(pixels, 1700000000.25, 12.5, 7)]
    track = VirtualVideoTrack(lambda: supplied[0], fps=1000)
    asyncio.run(track.recv())
    supplied[0] = CapturedVideoFrame(pixels, 1700000000.5, 12.75, 8)
    asyncio.run(track.recv())
    assert track.captured_frame_count == 2
    assert track._current_capture_wall_time == 1700000000.5
    assert track._current_capture_monotonic == 12.75


def test_placeholder_does_not_claim_camera_capture_or_sync_anchor():
    supplied = [None]
    track = VirtualVideoTrack(lambda: supplied[0], fps=1000, width=8, height=8)
    track.sync_frame_target = 0
    asyncio.run(track.recv())
    assert track.captured_frame_count == 0
    assert track._current_capture_wall_time is None
    assert track._current_capture_monotonic is None
    assert track.sync_frame_pts is None

    # The first actual image after a placeholder can anchor at its real sender
    # index rather than inventing a camera timestamp for placeholder frame 0.
    supplied[0] = np.zeros((8, 8, 3), dtype=np.uint8)
    asyncio.run(track.recv())
    assert track.sync_frame_target == track.sync_frame_pts == 1
    assert track.sync_frame_timestamp > 0


@pytest.mark.parametrize(
    "wall,mono,acquisition",
    [
        (float("nan"), 1.0, 0),
        (1.0, float("inf"), 0),
        (0.0, 1.0, 0),
        (1.0, -1.0, 0),
        (1.0, 1.0, -1),
        (1.0, 1.0, True),
    ],
)
def test_capture_sample_rejects_invalid_clock_and_identity(wall, mono, acquisition):
    from cyberwave.sensor.frame import CapturedVideoFrame

    with pytest.raises(ValueError):
        CapturedVideoFrame(np.zeros((8, 8, 3), dtype=np.uint8), wall, mono, acquisition)


def test_placeholder_omits_sei_then_real_sample_uses_existing_extension():
    sync = pytest.importorskip("cyberwave_video_sync")
    from cyberwave.sensor.frame import CapturedVideoFrame

    supplied = [None]
    track = VirtualVideoTrack(lambda: supplied[0], fps=1000, width=8, height=8)
    track._sync_enabled = True
    packets = [b"\x07test_sps", b"\x08test_pps", b"\x65test_idr"]
    asyncio.run(track.recv())
    assert sync.emit(track, packets, force_keyframe=True) == packets

    supplied[0] = CapturedVideoFrame(
        np.zeros((8, 8, 3), dtype=np.uint8), 1700000000.25, 12.5, 7
    )
    asyncio.run(track.recv())
    assert len(sync.emit(track, packets, force_keyframe=True)) == len(packets) + 1
    assert track._current_frame_index == 1
    assert track._current_capture_wall_time == 1700000000.25
    assert track._current_capture_monotonic == 12.5


def test_old_acquisition_does_not_claim_freshness_or_a_new_capture_clock():
    from cyberwave.sensor.frame import CapturedVideoFrame

    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    supplied = [CapturedVideoFrame(pixels, 1700000000.25, 12.5, 7)]
    track = VirtualVideoTrack(lambda: supplied[0], fps=1000)
    asyncio.run(track.recv())
    supplied[0] = CapturedVideoFrame(pixels, 1700000000.0, 12.25, 6)
    asyncio.run(track.recv())
    assert track.captured_frame_count == 1
    assert track._current_capture_wall_time is None
    assert track._current_capture_monotonic is None
