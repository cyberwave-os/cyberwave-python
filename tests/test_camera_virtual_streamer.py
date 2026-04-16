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

pytest.importorskip("aiortc", reason="aiortc not installed (install with extras: camera)")
sys.modules.setdefault(
    "yaml",
    types.SimpleNamespace(
        safe_load=lambda *args, **kwargs: {},
        dump=lambda *args, **kwargs: "",
    ),
)

from cyberwave.sensor.camera_virtual import VirtualCameraStreamer, VirtualVideoTrack


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
        streamer.streamer = SimpleNamespace(id="track-1", get_stream_attributes=lambda: {})

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
