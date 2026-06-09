"""Pin the per-stream ``stream_config`` wiring on the audio side.

CYB-2005 extends what CYB-2004 landed for the video side
(``BaseVideoStreamer`` + ``CV2CameraStreamer``) to the audio
publishers: ``BaseAudioStreamer`` previously didn't run
``EdgeHealthCheck`` at all, so paired microphone twins rendered
"Edge service not running" in the dashboard even when audio was
streaming.  These tests pin three properties of the fix:

1. ``MicrophoneAudioTrack.get_stream_config`` returns a typed
   ``stream_config`` block with the right discriminator and the
   required audio fields (``sample_rate_hz``, ``channels``).
2. ``BaseAudioStreamer._start_health_check`` constructs
   ``EdgeHealthCheck`` with a ``stream_config_provider`` so the wire
   stays in sync with track state heartbeat-to-heartbeat instead of
   freezing on a startup snapshot.
3. The end-to-end heartbeat for a microphone streamer carries
   ``streams["stream"].stream_config.kind == "audio"`` and the audio
   fields the frontend renders verbatim.

The full WebRTC lifecycle is not exercised — that would require
aiortc setup, an SDP exchange, and a real socket, which is exactly
the kind of brittleness these tests are meant to avoid.  Instead we
exercise the health-check wiring in isolation and trust the WebRTC
codepath via the existing ``MicrophoneAudioStreamer`` integration
suite.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import pytest

pytest.importorskip("av", reason="pyav not installed")
pytest.importorskip("aiortc", reason="aiortc not installed")

from cyberwave.edge.health import EdgeHealthCheck  # noqa: E402
from cyberwave.sensor.microphone import (  # noqa: E402
    BaseAudioTrack,
    MicrophoneAudioStreamer,
    MicrophoneAudioTrack,
)


class _FakeMQTT:
    topic_prefix = ""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def publish(self, topic: str, payload: Dict[str, Any], qos: int = 0) -> None:
        del qos
        self.calls.append((topic, dict(payload)))


def _drain_one_publish_cycle(
    checker: EdgeHealthCheck,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Run one publish cycle in-thread, bypassing the background loop."""
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


# =============================================================================
# Track-level stream_config
# =============================================================================


class TestMicrophoneAudioTrackStreamConfig:
    def test_mono_microphone_serialises_one_channel(self) -> None:
        """``layout == "mono"`` → ``channels == 1`` on the wire.

        The frontend's ``StreamConfig`` for ``kind: "audio"`` expects an
        integer ``channels`` (1 or 2) so it can render ``mono`` /
        ``stereo`` labels.  Anything else would silently fall through
        the dual-read fallback to the asset spec.

        Also pins that ``source`` is intentionally absent from the
        audio block.  Across the rest of the SDK ``source`` is a
        device path / URL / ROS topic; a WebRTC microphone has no
        equivalent and publishing the codec name there would overload
        the field's cross-publisher semantics.  The audio validator
        permits the omission.
        """
        track = MicrophoneAudioTrack(
            get_audio=lambda: None,
            sample_rate=48000,
            layout="mono",
        )
        cfg = track.get_stream_config()
        assert cfg == {
            "kind": "audio",
            "sample_rate_hz": 48000,
            "channels": 1,
            "codec": "opus",
        }
        assert "source" not in cfg

    def test_stereo_microphone_serialises_two_channels(self) -> None:
        track = MicrophoneAudioTrack(
            get_audio=lambda: None,
            sample_rate=48000,
            layout="stereo",
        )
        cfg = track.get_stream_config()
        assert cfg is not None
        assert cfg["channels"] == 2

    def test_alternate_sample_rate_propagates(self) -> None:
        """A future low-bandwidth microphone at 16 kHz still serialises correctly.

        Although Opus normally pins 48 kHz, a custom subclass can
        override the sample rate — the wire shape must reflect what
        the track is actually producing, not a hardcoded default.
        """
        track = MicrophoneAudioTrack(
            get_audio=lambda: None,
            sample_rate=16000,
            layout="mono",
        )
        cfg = track.get_stream_config()
        assert cfg is not None
        assert cfg["sample_rate_hz"] == 16000

    def test_base_audio_track_returns_none(self) -> None:
        """Generic ``BaseAudioTrack`` keeps the legacy no-stream_config shape.

        Subclasses that haven't been touched by CYB-2005 must continue
        to publish heartbeats without a ``stream_config`` block; the
        provider treats ``None`` as "use the legacy shape", not as
        "emit an empty block".
        """

        class _MinimalTrack(BaseAudioTrack):
            async def recv(self):  # pragma: no cover - not exercised
                raise NotImplementedError

        assert _MinimalTrack().get_stream_config() is None


# =============================================================================
# Streamer ↔ EdgeHealthCheck wiring
# =============================================================================


class _RecordingHealthCheck:
    """Stub for ``EdgeHealthCheck`` capturing constructor kwargs."""

    instances: List["_RecordingHealthCheck"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        _RecordingHealthCheck.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def patched_health_check(monkeypatch: pytest.MonkeyPatch):
    """Replace ``EdgeHealthCheck`` in the microphone module for the test."""
    _RecordingHealthCheck.instances.clear()

    # ``BaseAudioStreamer._start_health_check`` imports lazily from
    # ``cyberwave.edge.health``, so patch at the source.
    from cyberwave.edge import health as health_module

    monkeypatch.setattr(health_module, "EdgeHealthCheck", _RecordingHealthCheck)
    return lambda: _RecordingHealthCheck.instances


def _make_streamer(**overrides: Any) -> MicrophoneAudioStreamer:
    return MicrophoneAudioStreamer(
        _FakeMQTT(),
        get_audio=lambda: None,
        twin_uuid=overrides.pop("twin_uuid", "twin-mic"),
        mic_name=overrides.pop("mic_name", "mic"),
        sample_rate=overrides.pop("sample_rate", 48000),
        layout=overrides.pop("layout", "stereo"),
        **overrides,
    )


def test_start_health_check_wires_stream_config_provider(
    patched_health_check: Callable[[], List[_RecordingHealthCheck]],
) -> None:
    """``_start_health_check`` constructs ``EdgeHealthCheck`` with the provider.

    Without this, a regression to the registered-once pattern would
    silently freeze the wire on the requested sample rate / channels
    forever.  The provider is what makes the heartbeat reflect runtime
    truth.
    """
    streamer = _make_streamer()

    streamer._start_health_check()

    instances = patched_health_check()
    assert len(instances) == 1
    kwargs = instances[0].kwargs
    assert "stream_config_provider" in kwargs
    assert kwargs["stream_config_provider"] == streamer._collect_stream_configs
    assert instances[0].started


def test_collect_stream_configs_returns_empty_before_track_init(
    patched_health_check: Callable[[], List[_RecordingHealthCheck]],
) -> None:
    """Before ``_setup_webrtc`` runs, ``self.streamer`` is ``None``.

    Returning an empty dict keeps the heartbeat alive on the legacy
    single-``streams.stream`` shape rather than emitting a malformed
    block that would fail the validator and silently drop.
    """
    streamer = _make_streamer()
    assert streamer.streamer is None
    assert streamer._collect_stream_configs() == {}


def test_collect_stream_configs_returns_audio_block_when_track_initialised(
    patched_health_check: Callable[[], List[_RecordingHealthCheck]],
) -> None:
    """Post-track-init the provider returns the audio config under ``"stream"``."""
    streamer = _make_streamer(sample_rate=48000, layout="stereo")
    streamer.streamer = streamer.initialize_track()

    snapshot = streamer._collect_stream_configs()

    assert set(snapshot) == {"stream"}
    assert snapshot["stream"]["kind"] == "audio"
    assert snapshot["stream"]["channels"] == 2
    assert snapshot["stream"]["sample_rate_hz"] == 48000


def test_collect_stream_configs_absorbs_track_exception(
    patched_health_check: Callable[[], List[_RecordingHealthCheck]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A buggy override of ``get_stream_config`` must not break the heartbeat."""
    streamer = _make_streamer()
    streamer.streamer = streamer.initialize_track()

    def _broken() -> Dict[str, Any]:
        raise RuntimeError("subclass override regression")

    monkeypatch.setattr(streamer.streamer, "get_stream_config", _broken)

    assert streamer._collect_stream_configs() == {}


def test_start_health_check_is_noop_without_twin_uuid(
    patched_health_check: Callable[[], List[_RecordingHealthCheck]],
) -> None:
    """No ``twin_uuid`` → no heartbeat (regression guard for the early-bootstrap case)."""
    streamer = _make_streamer(twin_uuid=None)

    streamer._start_health_check()

    assert patched_health_check() == []


def test_start_health_check_skipped_when_disabled(
    patched_health_check: Callable[[], List[_RecordingHealthCheck]],
) -> None:
    """``enable_health_check=False`` opts out of the heartbeat entirely.

    Same flag semantics as the video side: lets local debugging
    sessions skip polluting MQTT with ``edge_health`` traffic.
    """
    streamer = _make_streamer(enable_health_check=False)

    streamer._start_health_check()

    assert patched_health_check() == []


def test_stop_health_check_idempotent(
    patched_health_check: Callable[[], List[_RecordingHealthCheck]],
) -> None:
    """``stop`` after ``start`` must not raise even when called twice."""
    streamer = _make_streamer()
    streamer._start_health_check()
    streamer._stop_health_check()
    streamer._stop_health_check()


# =============================================================================
# End-to-end: heartbeat carries kind="audio" on the wire
# =============================================================================


def test_audio_heartbeat_carries_audio_stream_config() -> None:
    """A microphone streamer's heartbeat advertises ``kind: "audio"`` on the wire.

    This is the contract the dashboard reads from to render ``48 kHz ·
    stereo``.  Wiring the real ``EdgeHealthCheck`` (not the stub)
    here pins the integration: provider returns the right shape, the
    publisher emits it under the canonical ``"stream"`` key, and the
    legacy ``camera_config`` slot stays ``None`` because audio is not
    a camera.
    """
    streamer = _make_streamer(sample_rate=48000, layout="stereo")
    streamer.streamer = streamer.initialize_track()
    mqtt = _FakeMQTT()
    health = EdgeHealthCheck(
        mqtt_client=mqtt,
        twin_uuids=["twin-mic"],
        edge_id="twin-mic",
        stream_config_provider=streamer._collect_stream_configs,
    )

    _drain_one_publish_cycle(health)

    payload = mqtt.calls[0][1]
    cfg = payload["streams"]["stream"]["stream_config"]
    assert cfg["kind"] == "audio"
    assert cfg["sample_rate_hz"] == 48000
    assert cfg["channels"] == 2
    assert cfg["codec"] == "opus"
    assert payload["camera_config"] is None


def test_audio_heartbeat_picks_up_track_changes_between_cycles() -> None:
    """Mutating the track between heartbeats reaches the next wire payload.

    Pins the freshness contract on the audio side too: a driver that
    re-opens its mic at a different sample rate (e.g. switching from
    16 kHz pre-warmup to 48 kHz steady state) is reflected on the
    very next heartbeat, no re-registration required.
    """
    streamer = _make_streamer(sample_rate=48000, layout="mono")
    streamer.streamer = streamer.initialize_track()
    mqtt = _FakeMQTT()
    health = EdgeHealthCheck(
        mqtt_client=mqtt,
        twin_uuids=["twin-mic"],
        edge_id="twin-mic",
        stream_config_provider=streamer._collect_stream_configs,
    )

    _drain_one_publish_cycle(health)
    assert (
        mqtt.calls[0][1]["streams"]["stream"]["stream_config"]["sample_rate_hz"]
        == 48000
    )

    # Simulate the streamer reopening at a different rate.
    streamer.streamer = MicrophoneAudioTrack(
        get_audio=lambda: None,
        sample_rate=16000,
        layout="mono",
    )

    _drain_one_publish_cycle(health)
    assert (
        mqtt.calls[1][1]["streams"]["stream"]["stream_config"]["sample_rate_hz"]
        == 16000
    )


def test_track_recv_increments_frame_count() -> None:
    """The audio track bumps a monotone ``frame_count`` per emitted frame.

    Used by ``_monitor_frame_count`` as a "have new frames arrived
    since last poll?" edge detector — only the *change* between
    polls is forwarded to ``EdgeHealthCheck`` (via ``mark_alive``,
    not ``update_frame_count``), so the counter must increment on
    every ``recv`` even though no single increment hits the wire.
    See ``EdgeHealthCheck.mark_alive`` for why audio doesn't forward
    per-frame.
    """
    import asyncio

    track = MicrophoneAudioTrack(
        get_audio=lambda: bytes(1920),
        sample_rate=48000,
        layout="mono",
    )
    assert track.frame_count == 0

    async def _drive() -> None:
        await track.recv()
        await track.recv()

    asyncio.run(_drive())
    assert track.frame_count == 2


# =============================================================================
# mark_alive() semantics: liveness without ``fps`` / ``frames_sent`` pollution
# =============================================================================


def test_audio_heartbeat_does_not_inflate_frames_sent_or_fps() -> None:
    """Audio publishers don't pollute ``fps`` / ``frames_sent`` on the wire.

    Pins the CYB-2005 self-review fix: the previous implementation
    forwarded every audio frame to ``update_frame_count``, which
    surfaced ``fps: ~50.0`` and ``frames_sent: <packet count>`` on
    the wire — terminology-correct (Opus emits a frame every 20 ms)
    but operationally noise.  After the fix the streamer calls
    ``mark_alive`` instead, which keeps ``last_frame_time`` fresh
    without bumping the counter.

    The dashboard already hides these fields for audio rows, but raw
    MQTT subscribers and the ``app_twintelemetry`` analytics table
    would still record the misleading numbers.  Pin the cleaner
    wire here so a future "switch back to update_frame_count for
    audio" refactor is caught.
    """
    mqtt = _FakeMQTT()
    health = EdgeHealthCheck(
        mqtt_client=mqtt,
        twin_uuids=["twin-mic"],
        edge_id="twin-mic",
    )

    # Simulate one poll cycle observing many audio packets.
    for _ in range(73):
        health.mark_alive()

    _drain_one_publish_cycle(health)
    stream = mqtt.calls[0][1]["streams"]["stream"]
    assert stream["frames_sent"] == 0, (
        "mark_alive() must never bump frames_sent; audio publishers "
        "rely on this to keep the wire honest."
    )
    assert stream["fps"] == 0.0, (
        "mark_alive() must never produce a non-zero fps; audio sample "
        "rate lives in stream_config.sample_rate_hz, not here."
    )


def test_mark_alive_flips_connection_state_to_connected() -> None:
    """One ``mark_alive`` call is enough to clear the "new"/"disconnected" state.

    Even though ``mark_alive`` leaves ``frame_count == 0``, the
    derived health fields the dashboard reads (``is_stale``,
    ``connection_state``, ``ice_connection_state``) must reflect
    that data has been seen.  Without this, an audio twin would
    render as "disconnected" indefinitely because the old
    ``frame_count > 0`` heuristic never trips.
    """
    mqtt = _FakeMQTT()
    health = EdgeHealthCheck(
        mqtt_client=mqtt,
        twin_uuids=["twin-mic"],
        edge_id="twin-mic",
    )

    health.mark_alive()
    _drain_one_publish_cycle(health)

    stream = mqtt.calls[0][1]["streams"]["stream"]
    assert stream["connection_state"] == "connected"
    assert stream["ice_connection_state"] == "connected"
    assert stream["is_stale"] is False
    assert stream["is_healthy"] is True


def test_audio_monitor_calls_mark_alive_once_per_poll_not_per_frame() -> None:
    """``_monitor_frame_count`` coarsens to one ``mark_alive`` per poll cycle.

    The previous implementation looped
    ``for _ in range(current - last): update_frame_count()`` — fine
    for video where ``fps`` is meaningful but wasteful for audio
    (50 packets/s × 10 polls/s = 500 unnecessary attribute writes
    per second per mic).  After the fix the monitor calls
    ``mark_alive`` exactly once per poll that observes any new
    frames, regardless of how many frames arrived.

    Pinning this also doubles as a regression net against "future
    refactor accidentally re-introduces per-frame forwarding by
    switching the call site to ``update_frame_count``" — that test
    would catch it because both spies would record the wrong count.
    """
    import asyncio

    streamer = _make_streamer(sample_rate=48000, layout="mono")
    track = streamer.initialize_track()
    streamer.streamer = track

    # Spy on both signals so we can prove ``mark_alive`` won (and
    # ``update_frame_count`` lost) regardless of how many audio
    # frames arrived between polls.
    mark_alive_calls = 0
    update_frame_calls = 0

    class _SpyHealth:
        def mark_alive(self) -> None:
            nonlocal mark_alive_calls
            mark_alive_calls += 1

        def update_frame_count(self) -> None:  # pragma: no cover
            nonlocal update_frame_calls
            update_frame_calls += 1

    streamer._health_check = _SpyHealth()  # type: ignore[assignment]
    streamer._last_frame_count = 0
    track.frame_count = 50  # one second of Opus packets at 50 Hz

    async def _one_tick() -> None:
        # Run one polling tick of the monitor, then break the
        # ``while self._is_running or self.pc is not None`` loop.
        streamer._is_running = True
        streamer.pc = None  # type: ignore[assignment]
        task = asyncio.create_task(streamer._monitor_frame_count())
        # ``asyncio.sleep(0.1)`` inside the monitor lets one
        # iteration run before we trip the loop guard.
        await asyncio.sleep(0.15)
        streamer._is_running = False
        await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(_one_tick())

    assert mark_alive_calls == 1, (
        "Audio monitor must call mark_alive exactly once per poll "
        "regardless of frame backlog; got "
        f"{mark_alive_calls}.  If this rose to ~50, the per-frame "
        "forwarding regression is back."
    )
    assert update_frame_calls == 0, (
        "Audio monitor must NOT call update_frame_count — that "
        "would pollute fps/frames_sent on the wire.  Got "
        f"{update_frame_calls} call(s)."
    )
