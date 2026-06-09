"""Unit tests for cyberwave.edge.health.EdgeHealthCheck.

Focus on the ``host_metrics_provider`` callback wiring (the publisher
must merge the provider's dict into every payload, and a misbehaving
provider must not stop the heartbeat) plus the per-stream
``stream_config`` registration API the dashboard relies on to render
camera / audio / lidar metadata without falling back to the asset spec.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Tuple

import pytest

from cyberwave.edge.health import EdgeHealthCheck


class _FakeMQTT:
    """Capture every publish() call; emulates the real client's interface."""

    topic_prefix = ""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def publish(self, topic: str, payload: Dict[str, Any], qos: int = 0) -> None:
        del qos
        self.calls.append((topic, dict(payload)))


def _drain_one_publish_cycle(
    checker: EdgeHealthCheck,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Run a single publish cycle in-thread, bypassing the background loop.

    Reaching directly into the inner helpers keeps the test deterministic
    (no sleep races) while still exercising the merge logic.
    """
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


class TestHostMetricsProvider:
    def test_payload_unchanged_without_provider(self) -> None:
        mqtt = _FakeMQTT()
        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
            host_metrics_provider=None,
        )

        _drain_one_publish_cycle(checker)

        assert len(mqtt.calls) == 1
        _, payload = mqtt.calls[0]
        assert payload["type"] == "edge_health"
        # None of the host-pressure fields should leak in when no provider
        # is registered — keeping driver-container payloads truly minimal.
        for key in (
            "host_memory_percent",
            "host_memory_available_mb",
            "cpu_temp_c",
            "consecutive_critical",
            "watchdog_layers",
        ):
            assert key not in payload, f"{key} unexpectedly present"

    def test_payload_merges_provider_keys(self) -> None:
        mqtt = _FakeMQTT()
        provided = {
            "host_memory_percent": 64.2,
            "host_memory_available_mb": 1432.0,
            "cpu_temp_c": 58.7,
            "consecutive_critical": 0,
            "watchdog_layers": ["systemd", "hardware"],
        }
        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
            host_metrics_provider=lambda: dict(provided),
        )

        _drain_one_publish_cycle(checker)

        assert len(mqtt.calls) == 1
        _, payload = mqtt.calls[0]
        for key, value in provided.items():
            assert payload[key] == value
        # Pre-existing fields are not clobbered.
        assert payload["type"] == "edge_health"
        assert payload["edge_id"] == "edge-1"

    def test_provider_exception_does_not_stop_publish(self) -> None:
        mqtt = _FakeMQTT()

        def _boom() -> Dict[str, Any]:
            raise RuntimeError("simulated host metrics failure")

        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
            host_metrics_provider=_boom,
        )

        # The publisher must still issue exactly one publish call even
        # when the provider raises — otherwise a buggy resource monitor
        # would silently mark the edge offline by suppressing all
        # heartbeats.
        _drain_one_publish_cycle(checker)

        assert len(mqtt.calls) == 1
        _, payload = mqtt.calls[0]
        assert payload["type"] == "edge_health"
        # The merge must be effectively empty when the provider raised.
        assert "host_memory_percent" not in payload
        assert "watchdog_layers" not in payload

    def test_provider_returning_non_dict_is_ignored(self) -> None:
        """Defensive shape check: tolerate buggy providers that return e.g. None."""
        mqtt = _FakeMQTT()
        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
            host_metrics_provider=lambda: None,  # type: ignore[arg-type, return-value]
        )

        _drain_one_publish_cycle(checker)

        assert len(mqtt.calls) == 1
        _, payload = mqtt.calls[0]
        assert "host_memory_percent" not in payload

    def test_provider_invoked_per_publish_cycle(self) -> None:
        """The provider must be re-invoked each publish, not memoised."""
        mqtt = _FakeMQTT()
        calls: List[float] = []

        def _provider() -> Dict[str, Any]:
            calls.append(time.time())
            return {"host_memory_percent": float(len(calls))}

        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
            host_metrics_provider=_provider,
        )

        _drain_one_publish_cycle(checker)
        _drain_one_publish_cycle(checker)
        _drain_one_publish_cycle(checker)

        assert len(calls) == 3
        # Each cycle saw a fresh value (1.0 → 2.0 → 3.0).
        assert [c[1]["host_memory_percent"] for c in mqtt.calls] == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "metrics",
    [
        {"host_memory_percent": 95.0},  # critical-only
        {"cpu_temp_c": 89.0},  # critical-only
        {"watchdog_layers": []},  # empty list still must be transmitted as-is
    ],
)
def test_partial_provider_payloads_preserved(metrics: Dict[str, Any]) -> None:
    """Provider may emit any subset of keys; the publisher must pass them through.

    This protects against future shape changes on either side: as long as
    edge-core and the SDK agree on additive optional keys, the dashboard
    only needs to update its display logic, not the wire contract.
    """
    mqtt = _FakeMQTT()
    checker = EdgeHealthCheck(
        mqtt_client=mqtt,
        twin_uuids=["twin-a"],
        edge_id="edge-1",
        host_metrics_provider=lambda: dict(metrics),
    )

    _drain_one_publish_cycle(checker)

    assert len(mqtt.calls) == 1
    _, payload = mqtt.calls[0]
    for key, value in metrics.items():
        assert payload[key] == value


# ---------------------------------------------------------------------------
# Per-stream stream_config registration
# ---------------------------------------------------------------------------


class TestRegisterStreamConfig:
    """Pin the public contract drivers (and CYB-2005) will consume.

    These tests fix the API surface of ``register_stream_config`` so a
    refactor cannot silently change field names or merge semantics: every
    downstream driver (camera_cv2, av_streamer, future microphone /
    lidar wiring) reads this contract.
    """

    def test_payload_without_register_and_no_frames_has_empty_streams(self) -> None:
        """Bootstrap-style publishers emit no phantom stream entry.

        A publisher that has never called ``register_stream_config``,
        has no ``stream_config_provider``, and has never seen a frame
        (``frame_count == 0``) is almost always the edge-core bootstrap
        publisher running in the gap before a driver container starts.
        Before CYB-2004 PR 2 we forced a single ``streams["stream"]``
        entry on the wire even in that case, which rendered as a
        misleading "0.0 fps" row in the dashboard's StreamsSection
        until the real driver took over.  The fix is to emit an empty
        ``streams`` map so the section renders nothing, and the legacy
        ``camera_config`` slot stays ``None``.
        """
        mqtt = _FakeMQTT()
        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )

        _drain_one_publish_cycle(checker)

        _, payload = mqtt.calls[0]
        assert payload["streams"] == {}
        assert payload["stream_count"] == 0
        assert payload["healthy_streams"] == 0
        # Legacy slot stays ``None`` when no camera-kind config is
        # registered — preserved for out-of-tree consumers that read
        # the deprecated top-level field.
        assert payload["camera_config"] is None

    def test_payload_without_register_keeps_single_stream_when_frames_flow(
        self,
    ) -> None:
        """Legacy drivers that drive ``update_frame_count`` keep their entry.

        ``camera_sim`` and the pre-CYB-2004 ``av_streamer`` path don't
        call ``register_stream_config``; they call ``update_frame_count``
        from the streamer's render loop.  Those drivers must still
        surface in the dashboard as soon as the first frame flows, so
        the historical single-stream wire shape (``streams["stream"]``
        with live fps + staleness) is preserved when ``frame_count > 0``.
        Only the truly idle case (no register, no frames) drops to an
        empty ``streams`` map.
        """
        mqtt = _FakeMQTT()
        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        checker.update_frame_count()

        _drain_one_publish_cycle(checker)

        _, payload = mqtt.calls[0]
        assert list(payload["streams"]) == ["stream"]
        assert payload["stream_count"] == 1
        stream_entry = payload["streams"]["stream"]
        # No ``stream_config`` block because no driver registered one —
        # only the live counters are present.
        assert "stream_config" not in stream_entry
        assert stream_entry["frames_sent"] == 1
        assert payload["camera_config"] is None

    def test_camera_kind_stream_config_inlined_into_streams_entry(self) -> None:
        """The registered block lands at ``streams[id].stream_config`` verbatim."""
        mqtt = _FakeMQTT()
        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        checker.register_stream_config(
            "stream",
            {
                "kind": "camera",
                "source": "/dev/video0",
                "resolution": "1280x720",
                "fps": 15,
                "camera_type": "cv2",
            },
        )

        _drain_one_publish_cycle(checker)

        _, payload = mqtt.calls[0]
        stream_config = payload["streams"]["stream"]["stream_config"]
        assert stream_config == {
            "kind": "camera",
            "source": "/dev/video0",
            "resolution": "1280x720",
            "fps": 15,
            "camera_type": "cv2",
        }
        # Per-stream live fields are not clobbered by the merge.
        assert "frames_sent" in payload["streams"]["stream"]
        assert "is_stale" in payload["streams"]["stream"]

    def test_camera_kind_populates_legacy_camera_config_shim(self) -> None:
        """During the deprecation window, camera configs mirror to ``camera_config``.

        Out-of-tree consumers that still read the top-level
        ``camera_config`` slot must see a populated block when a camera
        is publishing; only the new path requires reading
        ``streams[id].stream_config``.  This guarantee expires one
        release after CYB-2004 ships.
        """
        mqtt = _FakeMQTT()
        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        checker.register_stream_config(
            "stream",
            {
                "kind": "camera",
                "source": "/dev/video0",
                "resolution": "1280x720",
                "fps": 15,
                "camera_type": "cv2",
            },
        )

        _drain_one_publish_cycle(checker)

        _, payload = mqtt.calls[0]
        assert payload["camera_config"] == {
            "camera_id": "stream",
            "camera_type": "cv2",
            "source": "/dev/video0",
            "fps": 15,
            "resolution": "1280x720",
            "enabled": True,
        }

    @pytest.mark.parametrize(
        "kind, extra_fields",
        [
            ("audio", {"sample_rate_hz": 48000, "channels": 2}),
            ("lidar", {"scan_rate_hz": 10}),
            ("imu", {"rate_hz": 200}),
        ],
    )
    def test_non_camera_kind_leaves_camera_config_null(
        self, kind: str, extra_fields: Dict[str, Any]
    ) -> None:
        """Microphone / lidar / IMU configs must not leak into ``camera_config``.

        A future microphone publisher registering an ``audio`` config
        would otherwise corrupt the camera-only legacy field — exactly
        the kind of silent type confusion the discriminated union exists
        to prevent.  The kind-specific extras (``sample_rate_hz`` for
        audio, ``scan_rate_hz`` for lidar, ``rate_hz`` for imu) are
        required by the validator and are the same fields the frontend
        would render — so the test doubles as a contract pin.
        """
        mqtt = _FakeMQTT()
        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        checker.register_stream_config(
            "stream",
            {"kind": kind, "source": "/dev/null", **extra_fields},
        )

        _drain_one_publish_cycle(checker)

        _, payload = mqtt.calls[0]
        assert payload["streams"]["stream"]["stream_config"]["kind"] == kind
        assert payload["camera_config"] is None

    def test_legacy_camera_config_winner_is_deterministic(self) -> None:
        """Multi-camera devices pick the lexicographically-first stream id.

        A RealSense d455 publishing both ``depth-0`` and ``rgb-0`` would
        otherwise see the shim flap between cameras across heartbeats
        depending on dict iteration order, which is a UX regression
        operators notice as the resolution / source line changing every
        5 s.  Pinning the order ensures the shim is stable.
        """
        mqtt = _FakeMQTT()
        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        checker.register_stream_config(
            "rgb-0",
            {
                "kind": "camera",
                "source": "/dev/video0",
                "resolution": "1280x720",
                "fps": 30,
                "camera_type": "realsense",
            },
        )
        checker.register_stream_config(
            "depth-0",
            {
                "kind": "camera",
                "source": "/dev/video1",
                "resolution": "848x480",
                "fps": 15,
                "camera_type": "realsense-depth",
            },
        )

        _drain_one_publish_cycle(checker)

        _, payload = mqtt.calls[0]
        # ``depth-0`` < ``rgb-0`` alphabetically — that one wins.
        assert payload["camera_config"]["camera_id"] == "depth-0"
        assert payload["camera_config"]["camera_type"] == "realsense-depth"

    def test_register_is_idempotent(self) -> None:
        """Re-registering replaces, does not accumulate.

        Drivers may re-register on reconnect or after a config refresh;
        the latest call must win, with no leftover fields from the
        previous call hanging around in the merged dict.
        """
        mqtt = _FakeMQTT()
        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        checker.register_stream_config(
            "stream",
            {
                "kind": "camera",
                "source": "/dev/video0",
                "resolution": "640x480",
                "fps": 30,
                "camera_type": "cv2",
            },
        )
        checker.register_stream_config(
            "stream",
            {
                "kind": "camera",
                "source": "/dev/video1",
                "resolution": "1280x720",
                "fps": 15,
                "camera_type": "cv2",
            },
        )

        _drain_one_publish_cycle(checker)

        _, payload = mqtt.calls[0]
        cfg = payload["streams"]["stream"]["stream_config"]
        assert cfg["source"] == "/dev/video1"
        assert cfg["resolution"] == "1280x720"
        assert cfg["fps"] == 15

    def test_unregister_drops_stream_config(self) -> None:
        """Symmetric API: drivers can retract a config when tearing down.

        After unregister, with no frames seen and no other configs, the
        publisher falls back to the same empty-streams shape as the
        bootstrap publisher (CYB-2004 PR 2): an entry that was only
        announced via ``register_stream_config`` disappears entirely
        rather than leaving a stub row behind.  Drivers that called
        ``update_frame_count`` keep their legacy ``streams["stream"]``
        entry — see ``test_unregister_with_frames_keeps_legacy_entry``.
        """
        mqtt = _FakeMQTT()
        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        checker.register_stream_config(
            "stream",
            {
                "kind": "camera",
                "source": "/dev/video0",
                "resolution": "640x480",
                "fps": 30,
                "camera_type": "cv2",
            },
        )
        checker.unregister_stream_config("stream")

        _drain_one_publish_cycle(checker)

        _, payload = mqtt.calls[0]
        assert payload["streams"] == {}
        assert payload["camera_config"] is None

    def test_unregister_with_frames_keeps_legacy_entry(self) -> None:
        """Legacy compat: drivers that drive ``update_frame_count`` keep their row after unregister.

        A driver that registered a config, kept ticking ``frame_count``,
        and then unregistered must still surface its live counters under
        the legacy single-stream entry — that's the only way the
        dashboard can tell whether the unregistered stream is still
        flowing or has fully torn down.  The ``stream_config`` block
        is gone, but the entry remains.
        """
        mqtt = _FakeMQTT()
        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        checker.register_stream_config(
            "stream",
            {
                "kind": "camera",
                "source": "/dev/video0",
                "resolution": "640x480",
                "fps": 30,
                "camera_type": "cv2",
            },
        )
        checker.update_frame_count()
        checker.unregister_stream_config("stream")

        _drain_one_publish_cycle(checker)

        _, payload = mqtt.calls[0]
        assert list(payload["streams"]) == ["stream"]
        assert "stream_config" not in payload["streams"]["stream"]
        assert payload["camera_config"] is None

    def test_register_rejects_empty_stream_id(self) -> None:
        checker = EdgeHealthCheck(
            mqtt_client=_FakeMQTT(),
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        with pytest.raises(ValueError):
            checker.register_stream_config("", {"kind": "camera"})

    def test_register_rejects_non_dict_config(self) -> None:
        checker = EdgeHealthCheck(
            mqtt_client=_FakeMQTT(),
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        with pytest.raises(TypeError):
            checker.register_stream_config("stream", "not-a-dict")  # type: ignore[arg-type]

    def test_register_accepts_unknown_kind_as_additive(self) -> None:
        """Forward compatibility: unknown kinds pass through, not raise.

        The schema is intentionally additive so a future sensor kind
        (encoder, GPS, force-torque) can ship from a driver before the
        dashboard learns to render it.  Strict validation would couple
        every driver release to a coordinated frontend rollout — exactly
        what we want to avoid.
        """
        mqtt = _FakeMQTT()
        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        checker.register_stream_config(
            "stream",
            {"kind": "gps", "source": "/dev/ttyACM0", "rate_hz": 10},
        )

        _drain_one_publish_cycle(checker)

        _, payload = mqtt.calls[0]
        assert payload["streams"]["stream"]["stream_config"]["kind"] == "gps"
        assert payload["camera_config"] is None

    def test_register_does_not_share_dict_with_caller(self) -> None:
        """Mutating the caller's dict after register must not change the wire.

        Drivers commonly build a config dict, register it, then keep
        mutating it as part of their own state.  The publisher must
        snapshot at registration time so later mutations do not leak
        onto the wire — otherwise the dashboard sees inconsistent
        configs depending on driver internal state.
        """
        mqtt = _FakeMQTT()
        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        config = {
            "kind": "camera",
            "source": "/dev/video0",
            "resolution": "640x480",
            "fps": 30,
            "camera_type": "cv2",
        }
        checker.register_stream_config("stream", config)
        config["source"] = "/dev/video999"

        _drain_one_publish_cycle(checker)

        _, payload = mqtt.calls[0]
        assert payload["streams"]["stream"]["stream_config"]["source"] == "/dev/video0"

    def test_concurrent_register_does_not_corrupt_publish(self) -> None:
        """Pin the lock contract: registering during publish never raises.

        ``get_health_data`` is called on the publisher thread; drivers
        register from their own threads.  The lock guarantees the
        publisher always observes a consistent snapshot.  A bug here
        would show up as ``RuntimeError: dictionary changed size during
        iteration`` in production, which the lock prevents.
        """
        mqtt = _FakeMQTT()
        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        stop = threading.Event()

        def _spam_register() -> None:
            i = 0
            while not stop.is_set():
                checker.register_stream_config(
                    f"stream-{i % 4}",
                    {
                        "kind": "camera",
                        "source": "x",
                        "resolution": "1x1",
                        "fps": 1,
                        "camera_type": "cv2",
                    },
                )
                i += 1

        t = threading.Thread(target=_spam_register)
        t.start()
        try:
            for _ in range(50):
                _drain_one_publish_cycle(checker)
        finally:
            stop.set()
            t.join(timeout=2)

        # The exact count is timing-dependent; we only care that
        # nothing raised and every payload validates structurally.
        assert len(mqtt.calls) == 50
        for _, payload in mqtt.calls:
            assert payload["type"] == "edge_health"
            assert "streams" in payload


class TestRequiredFieldValidation:
    """Kind-specific required fields raise at registration time.

    Without these checks a driver could ship a half-built ``stream_config``
    (e.g. ``kind="camera"`` without ``resolution``) that the backend
    happily forwards as ``extra="allow"`` and the frontend renders as
    an empty row.  Catching at the registration boundary keeps the
    failure on the noisy side (driver crash on startup) rather than on
    the silent side (one broken row in a dashboard a customer reports
    days later).
    """

    @pytest.mark.parametrize(
        "kind, complete, missing_field",
        [
            (
                "camera",
                {"kind": "camera", "source": "0", "resolution": "640x480", "fps": 30},
                "resolution",
            ),
            (
                "camera",
                {"kind": "camera", "source": "0", "resolution": "640x480", "fps": 30},
                "fps",
            ),
            (
                "audio",
                {
                    "kind": "audio",
                    "source": "/dev/snd",
                    "sample_rate_hz": 48000,
                    "channels": 2,
                },
                "sample_rate_hz",
            ),
            (
                "audio",
                {
                    "kind": "audio",
                    "source": "/dev/snd",
                    "sample_rate_hz": 48000,
                    "channels": 2,
                },
                "channels",
            ),
            (
                "lidar",
                {"kind": "lidar", "source": "/dev/ttyUSB0", "scan_rate_hz": 10},
                "scan_rate_hz",
            ),
            (
                "imu",
                {"kind": "imu", "source": "/dev/ttyACM0", "rate_hz": 200},
                "rate_hz",
            ),
        ],
    )
    def test_raises_when_required_field_missing(
        self, kind: str, complete: Dict[str, Any], missing_field: str
    ) -> None:
        incomplete = {k: v for k, v in complete.items() if k != missing_field}
        checker = EdgeHealthCheck(
            mqtt_client=_FakeMQTT(),
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        with pytest.raises(ValueError, match=missing_field):
            checker.register_stream_config("stream", incomplete)

    def test_camera_missing_source_raises_with_clear_error(self) -> None:
        """The error message names the offending field so driver authors don't have to grep."""
        checker = EdgeHealthCheck(
            mqtt_client=_FakeMQTT(),
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        with pytest.raises(ValueError) as exc:
            checker.register_stream_config(
                "stream",
                {"kind": "camera", "resolution": "640x480", "fps": 30},
            )
        assert "source" in str(exc.value)

    def test_audio_accepts_missing_source(self) -> None:
        """``kind: "audio"`` deliberately does not require ``source``.

        Across the SDK ``source`` is a device path / URL / ROS topic
        (camera publishes ``/dev/video0``, lidar publishes ``/point_cloud2``).
        A WebRTC microphone has no equivalent: publishing the host
        ALSA / CoreAudio device path is a leak risk, and publishing
        the codec instead would overload the field's cross-publisher
        semantics.  This test pins the validator's allowance — if a
        future change tightens ``audio``'s required set to include
        ``source``, ``MicrophoneAudioStreamer`` would fail to register
        on startup and every paired microphone twin in the field
        would silently stop publishing heartbeats.
        """
        checker = EdgeHealthCheck(
            mqtt_client=_FakeMQTT(),
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        # Should NOT raise.
        checker.register_stream_config(
            "stream",
            {"kind": "audio", "sample_rate_hz": 48000, "channels": 2},
        )

    def test_audio_still_requires_sample_rate_and_channels(self) -> None:
        """Relaxing ``source`` must not weaken the other audio requirements.

        Audio rows render ``48 kHz · stereo`` from these two fields;
        without them the row collapses to an empty meta string.  The
        validator must keep enforcing them at the driver boundary so
        the dashboard never has to defensively render audio-without-rate.
        """
        checker = EdgeHealthCheck(
            mqtt_client=_FakeMQTT(),
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        with pytest.raises(ValueError, match="sample_rate_hz"):
            checker.register_stream_config(
                "stream",
                {"kind": "audio", "channels": 2},
            )
        with pytest.raises(ValueError, match="channels"):
            checker.register_stream_config(
                "stream",
                {"kind": "audio", "sample_rate_hz": 48000},
            )


class TestMarkAlive:
    """``mark_alive()`` — liveness without ``fps`` / ``frames_sent`` pollution.

    Companion path to ``update_frame_count()`` for kinds whose
    packetisation rate is not the right wire metric (audio's 50 Hz
    Opus packets, future IMU's 100/200/1000 Hz samples).  Pins the
    behavioural difference: same liveness signal, no counter
    inflation.
    """

    def test_mark_alive_does_not_increment_frame_count(self) -> None:
        """``frame_count`` stays at zero so ``frames_sent`` / ``fps`` stay honest.

        This is the whole reason ``mark_alive`` exists.  If a future
        refactor merges it back into ``update_frame_count`` the audio
        wire shape will silently regress to publishing
        ``frames_sent: <packet count>`` and ``fps: 50.0`` — both
        terminology-correct but operationally misleading.  Failing
        this test catches that.
        """
        checker = EdgeHealthCheck(
            mqtt_client=_FakeMQTT(),
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        for _ in range(100):
            checker.mark_alive()
        assert checker.frame_count == 0

    def test_mark_alive_updates_last_frame_time(self) -> None:
        """The staleness clock advances so ``is_stale`` clears as expected."""
        checker = EdgeHealthCheck(
            mqtt_client=_FakeMQTT(),
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        before = checker.last_frame_time
        # ``time.time()`` has at least µs resolution on the platforms
        # we ship to; one call is enough for monotonic ordering.
        checker.mark_alive()
        assert checker.last_frame_time >= before
        assert checker.last_frame_time > 0

    def test_mark_alive_flips_ice_connection_state_to_connected(self) -> None:
        """Without this the dashboard would render audio twins as 'new' forever.

        Before the CYB-2005 self-review fix ``ice_connection_state``
        was derived from ``frame_count > 0``.  Since ``mark_alive``
        doesn't bump ``frame_count``, that heuristic would never
        trip for audio publishers and the row would render with the
        "not yet handshaken" UI state regardless of how long data
        had been flowing.  Pin the new ``_has_alive_signal``-based
        derivation so the row renders ``connected`` after the first
        ``mark_alive``.

        The "before" snapshot follows CYB-2004 PR 2's bootstrap-phantom
        contract: with no registered config and no liveness signal,
        ``streams`` is empty rather than carrying a single
        ``streams["stream"]`` row in the ``"new"`` state — the
        dashboard reads that emptiness as "publisher exists but has
        nothing to report yet", which is the same UX as the old
        ``"new"`` row without the misleading "0.0 fps" line.
        """
        checker = EdgeHealthCheck(
            mqtt_client=_FakeMQTT(),
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        before = checker.get_health_data()
        assert before["streams"] == {}
        assert before["stream_count"] == 0

        checker.mark_alive()
        after = checker.get_health_data()
        assert after["streams"]["stream"]["ice_connection_state"] == "connected"
        assert after["streams"]["stream"]["frames_sent"] == 0
        assert after["streams"]["stream"]["fps"] == 0.0

    def test_update_frame_count_also_sets_alive_signal(self) -> None:
        """``update_frame_count`` callers don't have to dual-call ``mark_alive``.

        The video publishers (``CV2CameraStreamer``,
        ``BaseVideoStreamer``) use ``update_frame_count`` per
        emitted frame.  They must continue to flip
        ``ice_connection_state`` to ``connected`` on first call —
        otherwise we'd silently break video staleness reporting
        while landing audio's coarsening.
        """
        checker = EdgeHealthCheck(
            mqtt_client=_FakeMQTT(),
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        checker.update_frame_count()
        snapshot = checker.get_health_data()
        assert snapshot["streams"]["stream"]["ice_connection_state"] == "connected"
        assert snapshot["streams"]["stream"]["frames_sent"] == 1


class TestMultiStreamEmission:
    """Per-stream emission: one ``streams[…]`` entry per registered id."""

    def test_two_registered_streams_produce_two_entries(self) -> None:
        """RealSense-style RGB + depth: two registrations, two entries."""
        mqtt = _FakeMQTT()
        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        checker.register_stream_config(
            "rgb-0",
            {
                "kind": "camera",
                "source": "0",
                "resolution": "1280x720",
                "fps": 30,
                "camera_type": "cv2",
            },
        )
        checker.register_stream_config(
            "depth-0",
            {
                "kind": "camera",
                "source": "0",
                "resolution": "640x480",
                "fps": 30,
                "camera_type": "realsense",
            },
        )

        _drain_one_publish_cycle(checker)

        _, payload = mqtt.calls[0]
        assert set(payload["streams"]) == {"rgb-0", "depth-0"}
        assert payload["stream_count"] == 2
        # Both entries carry their own stream_config, distinguished by
        # the discriminator the dashboard needs to render correctly.
        rgb = payload["streams"]["rgb-0"]["stream_config"]
        depth = payload["streams"]["depth-0"]["stream_config"]
        assert rgb["resolution"] == "1280x720"
        assert depth["resolution"] == "640x480"
        assert depth["camera_type"] == "realsense"

    def test_legacy_drivers_get_stream_entry_once_frames_flow(self) -> None:
        """Untouched legacy publishers keep the single ``"stream"`` shape — once frames flow.

        Back-compat for camera_sim / pre-CYB-2004 av_streamer / older
        third-party SDK forks: as soon as the first frame ticks
        ``frame_count``, the publisher emits the historical single-stream
        wire shape so the frontend doesn't have to special-case missing
        streams.  Before the first frame those publishers look just like
        the bootstrap publisher (no register, no frames) and emit an
        empty ``streams`` map — see
        ``TestRegisterStreamConfig.test_payload_without_register_and_no_frames_has_empty_streams``.
        """
        mqtt = _FakeMQTT()
        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
        )
        checker.update_frame_count()

        _drain_one_publish_cycle(checker)

        _, payload = mqtt.calls[0]
        assert list(payload["streams"]) == ["stream"]
        assert payload["stream_count"] == 1
        assert "stream_config" not in payload["streams"]["stream"]


class TestStreamConfigProvider:
    """The dynamic ``stream_config_provider`` callback wiring."""

    def test_provider_dict_appears_in_payload(self) -> None:
        """Provider's return value flows through to ``streams[…].stream_config``."""
        mqtt = _FakeMQTT()
        snapshot = [
            {
                "stream": {
                    "kind": "camera",
                    "source": "0",
                    "resolution": "640x480",
                    "fps": 12.5,
                    "camera_type": "cv2",
                }
            }
        ]
        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
            stream_config_provider=lambda: snapshot[0],
        )

        _drain_one_publish_cycle(checker)

        _, payload = mqtt.calls[0]
        assert payload["streams"]["stream"]["stream_config"]["fps"] == 12.5

    def test_provider_invoked_per_cycle_picks_up_changes(self) -> None:
        """The dynamic path is the runtime truth — mutate, drain, observe.

        Pins the freshness contract: a driver bumping ``actual_fps``
        post-negotiation reaches the wire on the very next heartbeat,
        no re-registration required.
        """
        mqtt = _FakeMQTT()
        state = {
            "stream": {
                "kind": "camera",
                "source": "0",
                "resolution": "640x480",
                "fps": 30,
                "camera_type": "cv2",
            }
        }
        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
            stream_config_provider=lambda: state,
        )
        _drain_one_publish_cycle(checker)
        state["stream"]["fps"] = 15.0
        _drain_one_publish_cycle(checker)

        first_fps = mqtt.calls[0][1]["streams"]["stream"]["stream_config"]["fps"]
        second_fps = mqtt.calls[1][1]["streams"]["stream"]["stream_config"]["fps"]
        assert first_fps == 30
        assert second_fps == 15.0

    def test_provider_value_overrides_registered_static(self) -> None:
        """When both APIs disagree the dynamic wins — runtime truth beats stale registration."""
        mqtt = _FakeMQTT()
        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
            stream_config_provider=lambda: {
                "stream": {
                    "kind": "camera",
                    "source": "0",
                    "resolution": "640x480",
                    "fps": 7,
                    "camera_type": "cv2",
                }
            },
        )
        checker.register_stream_config(
            "stream",
            {
                "kind": "camera",
                "source": "0",
                "resolution": "640x480",
                "fps": 30,
                "camera_type": "cv2",
            },
        )

        _drain_one_publish_cycle(checker)

        _, payload = mqtt.calls[0]
        assert payload["streams"]["stream"]["stream_config"]["fps"] == 7

    def test_provider_exception_does_not_stop_publish(self) -> None:
        """A misbehaving provider must not silently mark the edge offline."""
        mqtt = _FakeMQTT()

        def _broken() -> Dict[str, Any]:
            raise RuntimeError("driver bug")

        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
            stream_config_provider=_broken,
        )

        _drain_one_publish_cycle(checker)

        _, payload = mqtt.calls[0]
        assert payload["type"] == "edge_health"
        assert "streams" in payload
        # Fall-through path: no static registration, provider raised,
        # no frames seen → empty ``streams`` map (same shape as the
        # bootstrap publisher).  The heartbeat itself still flows so
        # the dashboard knows the edge is alive; it just doesn't
        # render a misleading phantom row for a stream that isn't
        # actually being published.  See CYB-2004 PR 2.
        assert payload["streams"] == {}

    def test_provider_invalid_kind_payload_falls_back_to_registered(self) -> None:
        """A provider that emits an invalid kind-specific block is ignored for that stream.

        The static registration remains the source of truth, so the
        dashboard sees the most recently-known-good config instead of
        flipping to garbage when the provider regresses.
        """
        mqtt = _FakeMQTT()
        checker = EdgeHealthCheck(
            mqtt_client=mqtt,
            twin_uuids=["twin-a"],
            edge_id="edge-1",
            stream_config_provider=lambda: {
                "stream": {"kind": "camera", "source": "0"},  # missing res + fps
            },
        )
        checker.register_stream_config(
            "stream",
            {
                "kind": "camera",
                "source": "0",
                "resolution": "640x480",
                "fps": 30,
                "camera_type": "cv2",
            },
        )

        _drain_one_publish_cycle(checker)

        _, payload = mqtt.calls[0]
        cfg = payload["streams"]["stream"]["stream_config"]
        assert cfg["resolution"] == "640x480"
        assert cfg["fps"] == 30
