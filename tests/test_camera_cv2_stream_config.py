"""End-to-end tests for ``CV2CameraStreamer._build_stream_config``.

Pins the camera-side wiring of the per-stream ``stream_config`` schema
introduced in CYB-2004.  The contract being protected here is:

1. A cv2 camera streamer advertises ``kind == "camera"`` plus the
   resolution / fps / source / camera_type fields the dashboard renders.
2. The block round-trips through ``EdgeHealthCheck.register_stream_config``
   into the wire payload at ``streams["stream"].stream_config``.
3. The legacy top-level ``camera_config`` slot is populated from the
   same source for one deprecation release.

Without these, a cv2 webcam regression would surface as a missing
``resolution · fps · cv2`` label on the Edge Details pane, which the
asset-spec fallback would silently mask.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

pytest.importorskip("cv2", reason="OpenCV not installed")
pytest.importorskip("av", reason="pyav not installed")

from cyberwave.edge.health import EdgeHealthCheck  # noqa: E402
from cyberwave.sensor.camera_cv2 import CV2CameraStreamer  # noqa: E402
from cyberwave.sensor.config import Resolution  # noqa: E402


class _FakeMQTT:
    """Minimal MQTT client stand-in matching the publish contract."""

    topic_prefix = ""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def publish(self, topic: str, payload: Dict[str, Any], qos: int = 0) -> None:
        del qos
        self.calls.append((topic, dict(payload)))


def _make_streamer(
    *,
    camera_id: Any = 0,
    fps: int = 15,
    resolution: Any = Resolution.HD,
) -> CV2CameraStreamer:
    """Construct CV2CameraStreamer without touching OpenCV capture state.

    The constructor only stores attributes; no camera is opened until
    ``start()`` is called, which the tests deliberately avoid.
    """
    return CV2CameraStreamer(
        client=_FakeMQTT(),
        camera_id=camera_id,
        fps=fps,
        resolution=resolution,
        twin_uuid="twin-a",
    )


def test_build_stream_config_for_local_camera_index() -> None:
    """Local cameras serialise the integer id as a string ``source``.

    The frontend renders ``source`` verbatim in the Edge Details pane;
    coercing to ``str`` here keeps the JSON wire-side typed as a
    string regardless of whether the driver used an int or a path.
    """
    streamer = _make_streamer(camera_id=0, fps=15, resolution=Resolution.HD)

    config = streamer._build_stream_config()

    assert config == {
        "kind": "camera",
        "source": "0",
        "resolution": "1280x720",
        "fps": 15,
        "camera_type": "cv2",
    }


def test_build_stream_config_for_rtsp_url_preserves_source_string() -> None:
    """Credential-free RTSP / IP URLs pass through verbatim.

    No ``user:pass@`` segment means the mask is a no-op; the URL must
    survive the round-trip unchanged so operators can still identify the
    camera in the Edge Details pane.
    """
    streamer = _make_streamer(
        camera_id="rtsp://192.168.1.100:554/stream",
        fps=10,
        resolution=Resolution.VGA,
    )

    config = streamer._build_stream_config()

    assert config is not None
    assert config["source"] == "rtsp://192.168.1.100:554/stream"
    assert config["camera_type"] == "cv2"
    assert config["resolution"] == "640x480"


def test_build_stream_config_masks_rtsp_credentials() -> None:
    """RTSP URLs with embedded ``user:pass@`` are masked before the wire.

    The ``edge_health`` payload is broadcast over MQTT, persisted to
    ``app_twintelemetry`` by Vector, and cached in browser
    ``localStorage`` — none of those should ever carry plaintext IP
    camera credentials.  We mask at the driver layer so every
    downstream sink is safe by default.
    """
    streamer = _make_streamer(
        camera_id="rtsp://admin:hunter2@192.168.1.100:554/stream",
        fps=10,
        resolution=Resolution.VGA,
    )

    config = streamer._build_stream_config()

    assert config is not None
    assert "hunter2" not in config["source"]
    assert "admin" not in config["source"]
    assert config["source"] == "rtsp://***@192.168.1.100:554/stream"


def test_build_stream_config_masks_http_credentials() -> None:
    """HTTP IP cameras (snapshot URLs) get the same treatment as RTSP."""
    streamer = _make_streamer(
        camera_id="http://operator:p%40ssword@cam.local/snapshot.jpg",
        fps=5,
        resolution=Resolution.VGA,
    )

    config = streamer._build_stream_config()

    assert config is not None
    assert "operator" not in config["source"]
    assert "p%40ssword" not in config["source"]
    assert config["source"] == "http://***@cam.local/snapshot.jpg"


def test_build_stream_config_preserves_at_in_non_url_sources() -> None:
    """The mask anchors on ``://`` so plain paths with ``@`` are untouched.

    V4L by-id symlinks contain '@' in some kernel/udev configurations
    (`/dev/v4l/by-id/usb-Generic_FHD_Cam_Pid:1234@VideoUSB1-video-index0`).
    Masking those would break the source field and confuse operators.
    """
    streamer = _make_streamer(
        camera_id="/dev/v4l/by-id/usb-Vendor_Model@VideoUSB1-video-index0",
        fps=15,
        resolution=Resolution.VGA,
    )

    config = streamer._build_stream_config()

    assert config is not None
    assert config["source"] == (
        "/dev/v4l/by-id/usb-Vendor_Model@VideoUSB1-video-index0"
    )


def test_build_stream_config_for_tuple_resolution() -> None:
    """Tuple resolutions are normalised to ``"WxH"``.

    The frontend's ``EdgeCameraConfig.resolution`` is a string; the
    Resolution enum stringifies to ``"WxH"`` natively but raw tuples
    coming from ``CameraConfig`` legacy callers do not.  Both have to
    serialise the same way on the wire so the Edge Details pane row
    never shows a Python-repr like ``"(800, 600)"``.
    """
    streamer = _make_streamer(camera_id=1, fps=20, resolution=(800, 600))

    config = streamer._build_stream_config()

    assert config is not None
    assert config["resolution"] == "800x600"


def test_stream_config_round_trips_into_edge_health_payload() -> None:
    """Full integration: streamer config → EdgeHealthCheck → wire payload.

    Catches a class of bug where the hook is correct but the registration
    path silently drops it — the dashboard would then look fine in unit
    tests but render the asset-spec fallback on real edges.
    """
    streamer = _make_streamer(camera_id=0, fps=15, resolution=Resolution.HD)
    mqtt = _FakeMQTT()
    health = EdgeHealthCheck(
        mqtt_client=mqtt,
        twin_uuids=["twin-a"],
        edge_id="edge-1",
    )
    config = streamer._build_stream_config()
    assert config is not None
    health.register_stream_config("stream", config)

    base_payload = {
        "type": "edge_health",
        "timestamp": 1700000000.0,
        "edge_id": health.edge_id,
        "uptime_seconds": 0.0,
        **health.get_health_data(),
    }
    payload = dict(base_payload, twin_uuid="twin-a")

    assert payload["streams"]["stream"]["stream_config"] == config
    # The legacy alias mirrors the same source/fps/resolution so
    # downstream consumers that still read the top-level slot see
    # consistent data during the deprecation window.
    assert payload["camera_config"] == {
        "camera_id": "stream",
        "camera_type": "cv2",
        "source": "0",
        "fps": 15,
        "resolution": "1280x720",
        "enabled": True,
    }


class _StubTrack:
    """Minimal stand-in for ``CV2VideoTrack`` exposing ``actual_fps`` only.

    The cv2 capture stack only learns the actual rate once V4L2 has
    negotiated, so ``actual_fps`` starts as ``None`` and becomes a
    positive float later.  The streamer reads it through
    ``self.streamer.actual_fps`` (where ``self.streamer`` is the track),
    so the stub mirrors that shape.
    """

    def __init__(self, actual_fps: Any) -> None:
        self.actual_fps = actual_fps


def test_build_stream_config_falls_back_to_requested_fps_before_negotiation() -> None:
    """Before the first frame ``actual_fps`` is ``None`` and we ship the request.

    There's nothing better to publish at this point — the dashboard's
    ``X fps`` row would otherwise be empty for the first heartbeat,
    which is the worst time to look uninitialised (it's exactly when
    operators are inspecting the edge to see if it came up).
    """
    streamer = _make_streamer(camera_id=0, fps=30, resolution=Resolution.HD)
    streamer.streamer = None  # explicit: pre-start state

    config = streamer._build_stream_config()

    assert config is not None
    assert config["fps"] == 30


def test_build_stream_config_publishes_negotiated_fps_when_available() -> None:
    """Once V4L2 has clamped the rate, the wire reflects the truth.

    Requesting 30 fps from a webcam that only delivers 15 produces
    ``actual_fps=15.0`` on the track.  The dashboard must show 15 (the
    truth) not 30 (the request); otherwise the ``fps · resolution``
    row lies and operators can't tell that the camera is degraded.
    """
    streamer = _make_streamer(camera_id=0, fps=30, resolution=Resolution.HD)
    streamer.streamer = _StubTrack(actual_fps=15.0)

    config = streamer._build_stream_config()

    assert config is not None
    assert config["fps"] == 15.0


def test_build_stream_config_ignores_non_positive_actual_fps() -> None:
    """``actual_fps=0`` is a not-yet-known sentinel, not a valid rate.

    Some backends initialise ``actual_fps`` to 0.0 before the first
    successful frame.  Publishing ``fps=0`` would make the dashboard
    show ``0 fps`` and the stream as degraded even though the streamer
    just hasn't measured yet — the requested rate is the right
    fallback.
    """
    streamer = _make_streamer(camera_id=0, fps=24, resolution=Resolution.HD)
    streamer.streamer = _StubTrack(actual_fps=0.0)

    config = streamer._build_stream_config()

    assert config is not None
    assert config["fps"] == 24


def test_stream_config_provider_path_propagates_dynamic_fps() -> None:
    """End-to-end: provider → EdgeHealthCheck → wire reflects current fps.

    Wires the streamer's ``_collect_stream_configs`` adapter through
    ``EdgeHealthCheck.stream_config_provider`` exactly as
    ``BaseVideoStreamer._start_health_check`` does, then mutates the
    streamer's ``actual_fps`` between heartbeats and confirms the
    second payload picks up the change.  Pins the freshness contract
    that the provider exists to enforce — without it the registered
    snapshot would stay frozen at the requested fps forever.
    """
    streamer = _make_streamer(camera_id=0, fps=30, resolution=Resolution.HD)
    streamer.streamer = _StubTrack(actual_fps=None)
    mqtt = _FakeMQTT()
    health = EdgeHealthCheck(
        mqtt_client=mqtt,
        twin_uuids=["twin-a"],
        edge_id="edge-1",
        stream_config_provider=streamer._collect_stream_configs,
    )

    first = health.get_health_data()
    assert first["streams"]["stream"]["stream_config"]["fps"] == 30

    streamer.streamer.actual_fps = 12.5

    second = health.get_health_data()
    assert second["streams"]["stream"]["stream_config"]["fps"] == 12.5
