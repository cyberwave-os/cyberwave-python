"""An edge device must come back on its own after its uplink drops.

Covers the SDK pieces a Go2 on a Jetson relies on: REST calls that cannot hang
on a half-open connection, offers that fail fast while MQTT is down, and an
audio reconnect that doesn't leak a heartbeat per attempt.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from cyberwave import Cyberwave
from cyberwave.client import _default_request_timeout
from cyberwave.config import CyberwaveConfig
from cyberwave.sensor._reconnect import (
    FAST_RECONNECT_ATTEMPTS,
    SLOW_RECONNECT_DELAY,
    reconnect_backoff,
)


def test_default_request_timeout_bounds_connect_and_read() -> None:
    assert _default_request_timeout(30) == (10.0, 30.0)
    assert _default_request_timeout(5) == (5.0, 5.0)
    assert _default_request_timeout(0) is None
    assert _default_request_timeout(None) is None


def test_a_call_with_a_body_gets_no_connect_timeout() -> None:
    # urllib3 keeps the connect timeout on the socket while it sends the body,
    # so a slow upload would be cut off after it.
    assert _default_request_timeout(30, has_body=True) == (None, 30.0)
    assert _default_request_timeout(0, has_body=True) is None


def test_api_timeout_env_overrides_the_default(monkeypatch) -> None:
    monkeypatch.setenv("CYBERWAVE_API_TIMEOUT", "12")
    assert CyberwaveConfig().timeout == 12
    # An explicit value wins over the environment, even one equal to the default.
    assert CyberwaveConfig(timeout=7).timeout == 7
    assert CyberwaveConfig(timeout=300).timeout == 300


def test_api_timeout_env_accepts_fractional_seconds(monkeypatch) -> None:
    monkeypatch.setenv("CYBERWAVE_API_TIMEOUT", "30.5")
    assert CyberwaveConfig().timeout == 30.5


@pytest.mark.parametrize("value", ["-5", "soon", "nan", "inf"])
def test_an_unusable_api_timeout_env_keeps_the_default(
    monkeypatch, caplog, value
) -> None:
    monkeypatch.setenv("CYBERWAVE_API_TIMEOUT", value)
    assert CyberwaveConfig().timeout == 300
    assert "Ignoring CYBERWAVE_API_TIMEOUT" in caplog.text


def test_the_default_read_timeout_leaves_room_for_slow_calls(monkeypatch) -> None:
    monkeypatch.delenv("CYBERWAVE_API_TIMEOUT", raising=False)
    assert CyberwaveConfig().timeout == 300


def _list_assets(client: Cyberwave, **extra) -> MagicMock:
    import inspect

    request = MagicMock(return_value=MagicMock())
    client._api_client.rest_client.request = request
    sig = inspect.signature(client.api._src_app_api_assets_list_assets_serialize)
    kwargs: dict = {
        "_request_auth": None,
        "_content_type": None,
        "_headers": None,
        "_host_index": 0,
    }
    for name, param in sig.parameters.items():
        if name not in kwargs and param.default is inspect.Parameter.empty:
            kwargs[name] = None
    serialized = client.api._src_app_api_assets_list_assets_serialize(**kwargs)
    client._api_client.call_api(*serialized, **extra)
    return request


def test_rest_calls_get_the_configured_timeout_by_default(monkeypatch) -> None:
    monkeypatch.setenv("CYBERWAVE_API_TIMEOUT", "30")
    client = Cyberwave(base_url="http://localhost:8000", api_key="k")
    request = _list_assets(client)
    assert request.call_args.kwargs["_request_timeout"] == (10.0, 30.0)


@pytest.mark.parametrize(
    "body, post_params",
    [({"name": "x"}, None), (b"glb-bytes", None), (None, [("file", ("a.glb", b"x"))])],
)
def test_uploads_and_json_bodies_get_no_connect_timeout(
    monkeypatch, body, post_params
) -> None:
    monkeypatch.setenv("CYBERWAVE_API_TIMEOUT", "30")
    client = Cyberwave(base_url="http://localhost:8000", api_key="k")
    request = MagicMock(return_value=MagicMock())
    client._api_client.rest_client.request = request

    client._api_client.call_api(
        "POST", "http://localhost:8000/api/v1/assets", {}, body, post_params
    )

    assert request.call_args.kwargs["_request_timeout"] == (None, 30.0)


def test_an_explicit_request_timeout_wins() -> None:
    client = Cyberwave(base_url="http://localhost:8000", api_key="k")
    request = _list_assets(client, _request_timeout=2.0)
    assert request.call_args.kwargs["_request_timeout"] == 2.0


def test_reconnect_backoff_slows_down_but_never_stops() -> None:
    delays = [reconnect_backoff(n) for n in range(1, FAST_RECONNECT_ATTEMPTS + 5)]
    assert delays == sorted(delays)
    assert delays[-1] == SLOW_RECONNECT_DELAY


pytest.importorskip("av", reason="pyav not installed")
pytest.importorskip("aiortc", reason="aiortc not installed")

from cyberwave.sensor import base_video, microphone  # noqa: E402


def _client(connected: bool) -> MagicMock:
    client = MagicMock()
    client.topic_prefix = ""
    client.client_id = "test-client"
    client.connected = connected
    # What CyberwaveMQTTClient.publish returns when it is not connected.
    client.publish.return_value = connected
    return client


def test_a_refused_video_offer_fails_the_attempt() -> None:
    class _Camera(base_video.BaseVideoStreamer):
        def initialize_track(self):  # pragma: no cover
            raise NotImplementedError

    camera = _Camera(client=_client(False), twin_uuid="t", enable_health_check=False)
    assert camera._publish_message("topic", {}) is False


def test_a_refused_audio_offer_raises() -> None:
    mic = microphone.MicrophoneAudioStreamer(
        _client(False),
        get_audio=lambda: None,
        twin_uuid="t",
        mic_name="mic",
        enable_health_check=False,
    )
    mic.pc = MagicMock()
    mic.pc.localDescription = MagicMock(sdp="v=0", type="offer")
    with pytest.raises(ConnectionError, match="refused the offer"):
        mic._send_offer("v=0")


def test_a_publish_override_returning_none_still_sends_the_offer() -> None:
    # Subclasses written to the old ``-> None`` contract must keep working.
    class _Mic(microphone.MicrophoneAudioStreamer):
        def _publish_message(self, topic, payload):
            sent.append(topic)

    sent: list = []
    mic = _Mic(
        _client(True),
        get_audio=lambda: None,
        twin_uuid="t",
        mic_name="mic",
        enable_health_check=False,
    )
    mic.pc = MagicMock()
    mic.pc.localDescription = MagicMock(sdp="v=0", type="offer")
    mic._send_offer("v=0")
    assert len(sent) == 1

@pytest.mark.asyncio
async def test_closing_the_audio_peer_stops_its_heartbeat() -> None:
    mic = microphone.MicrophoneAudioStreamer(
        _client(True), get_audio=lambda: None, twin_uuid="t", mic_name="mic"
    )
    heartbeat = MagicMock()
    monitor = asyncio.get_running_loop().create_future()
    mic._health_check = heartbeat
    mic._health_monitor_task = monitor

    await mic._close_peer_connection()

    heartbeat.stop.assert_called_once()
    assert monitor.cancelled()
    assert mic._health_check is None
