"""Pin ``MultimediaStreamer``'s audio ``stream_config`` wiring.

Companion to ``test_microphone_health_wiring.py``: ``av_streamer.py``
manages its own WebRTC peer connection with paired video + audio
tracks, so the per-stream config wiring is duplicated from
``BaseAudioStreamer``.  These tests pin the duplication doesn't drift —
specifically that the multimedia streamer constructs ``EdgeHealthCheck``
with a ``stream_config_provider`` that returns the audio track's
config under the ``"audio"`` key.

Audio is keyed as ``"audio"`` (not the legacy ``"stream"`` placeholder
that single-stream publishers use) because ``MultimediaStreamer``'s
whole point is paired video + audio — the moment ``BaseVideoTrack``
gets a ``get_stream_config`` hook, the provider should be able to add
``result["video"]`` next to the existing ``result["audio"]`` without
a wire-breaking rename.  Pinning this key here keeps the multi-stream
contract enforceable from CYB-2005 onward.

Video is out of scope for CYB-2005 (no ``get_stream_config`` hook on
the generic ``BaseVideoTrack``); we assert that the provider returns
the audio-only block and the heartbeat picks it up.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

pytest.importorskip("av", reason="pyav not installed")
pytest.importorskip("aiortc", reason="aiortc not installed")

from cyberwave.edge.health import EdgeHealthCheck  # noqa: E402
from cyberwave.sensor.av_streamer import MultimediaStreamer  # noqa: E402
from cyberwave.sensor.microphone import MicrophoneAudioTrack  # noqa: E402


class _FakeMQTT:
    topic_prefix = ""
    client_id = "test-client"

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def publish(self, topic: str, payload: Dict[str, Any], qos: int = 0) -> None:
        del qos
        self.calls.append((topic, dict(payload)))


def _drain_one_publish_cycle(
    checker: EdgeHealthCheck,
) -> List[Tuple[str, Dict[str, Any]]]:
    base_payload = {
        "type": "edge_health",
        "timestamp": 1700000000.0,
        "edge_id": checker.edge_id,
        "uptime_seconds": 0.0,
        **checker.get_health_data(),
        **checker._collect_host_metrics(),
    }
    mqtt = checker.mqtt_client
    out: List[Tuple[str, Dict[str, Any]]] = []
    for twin_uuid in checker.twin_uuids:
        payload = dict(base_payload, twin_uuid=twin_uuid)
        topic = f"cyberwave/twin/{twin_uuid}/edge_health"
        mqtt.publish(topic, payload)
        out.append((topic, payload))
    return out


def _make_streamer() -> MultimediaStreamer:
    """Build a MultimediaStreamer without setting up WebRTC.

    The tests poke ``audio_track`` directly to exercise
    ``_collect_stream_configs``; the WebRTC lifecycle and offer
    plumbing live behind ``_start_webrtc`` and would require an SDP
    exchange to exercise — out of scope for a unit test.
    """
    return MultimediaStreamer(
        client=_FakeMQTT(),
        create_video_track=lambda: MagicMock(),
        create_audio_track=lambda: MagicMock(),
        twin_uuid="twin-av",
        camera_name="cam",
        mic_name="mic",
    )


class _RecordingHealthCheck:
    """Capture EdgeHealthCheck constructor kwargs for assertion."""

    instances: List["_RecordingHealthCheck"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = False
        _RecordingHealthCheck.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        pass


@pytest.fixture
def patched_health_check(monkeypatch: pytest.MonkeyPatch):
    _RecordingHealthCheck.instances.clear()
    from cyberwave.edge import health as health_module

    monkeypatch.setattr(health_module, "EdgeHealthCheck", _RecordingHealthCheck)
    # Suppress monitor task creation in sync test context.
    monkeypatch.setattr(
        "cyberwave.sensor.av_streamer.asyncio.create_task", lambda coro: coro.close()
    )
    return lambda: _RecordingHealthCheck.instances


def test_start_health_check_wires_stream_config_provider(
    patched_health_check: Callable[[], List[_RecordingHealthCheck]],
) -> None:
    """``EdgeHealthCheck`` is constructed with ``stream_config_provider``."""
    streamer = _make_streamer()

    streamer._start_health_check()

    instances = patched_health_check()
    assert len(instances) == 1
    kwargs = instances[0].kwargs
    assert "stream_config_provider" in kwargs
    assert kwargs["stream_config_provider"] == streamer._collect_stream_configs


def test_collect_stream_configs_empty_when_audio_track_uninitialised(
    patched_health_check: Callable[[], List[_RecordingHealthCheck]],
) -> None:
    """Pre-``_setup_webrtc`` the audio track is ``None``; provider returns ``{}``."""
    streamer = _make_streamer()
    assert streamer.audio_track is None
    assert streamer._collect_stream_configs() == {}


def test_collect_stream_configs_returns_audio_block_from_track(
    patched_health_check: Callable[[], List[_RecordingHealthCheck]],
) -> None:
    """When the audio track is a ``MicrophoneAudioTrack`` the provider returns its config.

    Pins the ``"audio"`` key — the CYB-2005 self-review fix that
    replaced the legacy ``"stream"`` placeholder so the wire is
    already prepared for ``{"video": …, "audio": …}`` when video
    grows a ``get_stream_config`` hook.  Stuffing audio into
    ``"stream"`` today would force a wire-breaking rename then.
    """
    streamer = _make_streamer()
    streamer.audio_track = MicrophoneAudioTrack(
        get_audio=lambda: None,
        sample_rate=48000,
        layout="mono",
    )

    snapshot = streamer._collect_stream_configs()

    assert set(snapshot) == {"audio"}, (
        "MultimediaStreamer must key audio under 'audio', not 'stream' — "
        "the latter is reserved for single-stream publishers; this "
        "streamer's whole point is paired audio + video."
    )
    assert snapshot["audio"]["kind"] == "audio"
    assert snapshot["audio"]["sample_rate_hz"] == 48000
    assert snapshot["audio"]["channels"] == 1


def test_collect_stream_configs_empty_when_track_returns_none(
    patched_health_check: Callable[[], List[_RecordingHealthCheck]],
) -> None:
    """Generic ``BaseAudioTrack`` (no override) preserves the legacy wire shape."""
    streamer = _make_streamer()
    fake_track = MagicMock()
    fake_track.get_stream_config.return_value = None
    streamer.audio_track = fake_track

    assert streamer._collect_stream_configs() == {}


def test_collect_stream_configs_absorbs_track_exception(
    patched_health_check: Callable[[], List[_RecordingHealthCheck]],
) -> None:
    """A broken ``get_stream_config`` override falls back to empty (keeps heartbeat alive)."""
    streamer = _make_streamer()
    fake_track = MagicMock()
    fake_track.get_stream_config.side_effect = RuntimeError("subclass regression")
    streamer.audio_track = fake_track

    assert streamer._collect_stream_configs() == {}


def test_audio_heartbeat_carries_audio_stream_config_end_to_end() -> None:
    """Real ``EdgeHealthCheck`` + real provider → wire carries the audio block.

    Pins the contract the dashboard reads from for multimedia twins:
    even though video and audio share one peer connection, the audio
    metadata flows through the per-stream channel — the dashboard
    doesn't have to crack the SDP for ``48 kHz · stereo``.
    """
    streamer = _make_streamer()
    streamer.audio_track = MicrophoneAudioTrack(
        get_audio=lambda: None,
        sample_rate=48000,
        layout="stereo",
    )
    mqtt = _FakeMQTT()
    health = EdgeHealthCheck(
        mqtt_client=mqtt,
        twin_uuids=["twin-av"],
        edge_id="twin-av",
        stream_config_provider=streamer._collect_stream_configs,
    )

    _drain_one_publish_cycle(health)

    # The streams map is keyed by the provider's keys — "audio" for
    # MultimediaStreamer, not "stream".  This is the wire-shape the
    # frontend asserts against in
    # ``edge-device-status-host-telemetry.test.tsx``.
    streams = mqtt.calls[0][1]["streams"]
    assert "audio" in streams, (
        f"Expected audio block under 'audio' key; got {list(streams)}"
    )
    cfg = streams["audio"]["stream_config"]
    assert cfg["kind"] == "audio"
    assert cfg["sample_rate_hz"] == 48000
    assert cfg["channels"] == 2
    # Audio twin must NOT pollute the legacy camera_config slot —
    # otherwise the dashboard's deprecated reader would mis-render
    # the row as a camera.
    assert mqtt.calls[0][1]["camera_config"] is None
