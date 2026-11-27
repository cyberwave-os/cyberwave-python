"""BaseVideoStreamer: consent and lag diagnostics, without changing reconnects."""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("aiortc", reason="aiortc not installed (install with extras: camera)")

from cyberwave.sensor import base_video  # noqa: E402
from cyberwave.sensor.base_video import BaseVideoStreamer  # noqa: E402
from cyberwave.sensor.ice_consent import ConsentWatch  # noqa: E402


class _Streamer(BaseVideoStreamer):
    def initialize_track(self):  # pragma: no cover - never called here
        raise NotImplementedError


def _streamer() -> _Streamer:
    client = MagicMock()
    client.client_id = "python-sdk_live_" + "a" * 32
    client.topic_prefix = ""
    return _Streamer(client, twin_uuid="twin-1", camera_name="head_cam")


def _lose_connection(streamer: _Streamer, caplog: pytest.LogCaptureFixture) -> str:
    streamer.pc = SimpleNamespace(connectionState="closed", iceConnectionState="closed")
    with caplog.at_level(logging.WARNING, logger="cyberwave.sensor.base_video"):
        for _ in range(base_video.CONNECTION_LOSS_CONFIRMATION_CHECKS - 1):
            assert streamer._is_connection_lost() is False
        assert streamer._is_connection_lost() is True
    return caplog.text


def test_lost_warning_says_the_close_followed_a_consent_expiry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    streamer = _streamer()
    streamer._consent_watch = ConsentWatch(
        label="twin twin-1 camera head_cam", tracked=True, expired_at=time.monotonic() - 2.6
    )
    text = _lose_connection(streamer, caplog)
    assert "WebRTC connection lost after 3 consecutive checks" in text
    assert "after ICE consent expired 2." in text


def test_lost_warning_says_when_no_consent_expiry_preceded_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    streamer = _streamer()
    streamer._consent_watch = ConsentWatch(label="twin twin-1 camera head_cam", tracked=True)
    assert "no ICE consent expiry or socket loss recorded" in _lose_connection(streamer, caplog)


def test_reconnect_timings_are_unchanged() -> None:
    assert base_video.CONNECTION_LOSS_CONFIRMATION_CHECKS == 3


def test_stream_label_names_twin_and_camera() -> None:
    assert _streamer()._stream_label() == "twin twin-1 camera head_cam"


async def test_stop_stops_the_lag_probe() -> None:
    streamer = _streamer()
    streamer._start_loop_lag_probe()
    probe = streamer._loop_lag_probe
    assert probe is not None and probe.running
    await streamer.stop()
    assert not probe.running
