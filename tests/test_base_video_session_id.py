"""The offer and every camera_sync_frame must share one session_id."""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("aiortc", reason="aiortc not installed (install with extras: camera)")

from cyberwave.sensor.base_video import BaseVideoStreamer  # noqa: E402


class _Streamer(BaseVideoStreamer):
    def initialize_track(self):  # pragma: no cover - never called here
        raise NotImplementedError


def _make_streamer() -> _Streamer:
    client = MagicMock()
    client.client_id = "python-sdk_live_" + "a" * 32
    client.topic_prefix = ""
    return _Streamer(client, twin_uuid="twin-1", camera_name="head_cam")


def test_camera_sync_frame_carries_the_offer_session_id(monkeypatch) -> None:
    streamer = _make_streamer()
    published = {}
    monkeypatch.setattr(
        streamer, "_publish_message", lambda topic, payload: published.update(payload)
    )
    streamer._publish_camera_sync_frame(
        frame_index=30,
        pts=3000,
        time_base_num=1,
        time_base_den=30,
        timestamp=1755300000.5,
        timestamp_monotonic=100.5,
    )
    assert published["session_id"] == "python-sdk_live_" + "a" * 32 + "_head_cam"
    assert published["sensor"] == "head_cam"
