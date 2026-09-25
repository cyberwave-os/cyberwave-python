"""Health liveness must follow captures, not encoder ticks.

The freeze/cached-frame fallbacks keep ``frame_count`` climbing while a camera
is unplugged, so a health check polling it reported a dead camera as the
healthiest stream on the device, indefinitely. Tracks with such a fallback
expose ``captured_frame_count`` instead. These tests pin both halves of that
protocol and their junction with ``EdgeHealthCheck``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

pytest.importorskip("cv2", reason="OpenCV not installed")
pytest.importorskip("av", reason="pyav not installed")

from cyberwave.edge.health import EdgeHealthCheck  # noqa: E402
from cyberwave.sensor.base_video import (  # noqa: E402
    SDK_EDGE_HEALTH_STALE_TIMEOUT_SECONDS,
    BaseVideoStreamer,
    read_liveness_counter,
)
from cyberwave.sensor.camera_cv2 import CV2CameraStreamer  # noqa: E402
from cyberwave.sensor.config import Resolution  # noqa: E402


class _FakeMQTT:
    topic_prefix = ""

    def __init__(self) -> None:
        self.calls: List[Any] = []

    def publish(self, topic: str, payload: Dict[str, Any], qos: int = 0) -> None:
        del qos
        self.calls.append((topic, dict(payload)))


class _FakeTrack:
    """Stand-in for a video track with independent emit/capture counters."""

    def __init__(self, captured: Optional[int] = 0) -> None:
        self.frame_count = 0
        if captured is not None:
            self.captured_frame_count = captured

    def emit(self, n: int = 1) -> None:
        """Frames handed to the encoder — freeze frames included."""
        self.frame_count += n

    def capture(self, n: int = 1) -> None:
        """Frames that actually came off the device."""
        self.frame_count += n
        self.captured_frame_count += n


def _health(
    stale_timeout: int = SDK_EDGE_HEALTH_STALE_TIMEOUT_SECONDS,
) -> EdgeHealthCheck:
    return EdgeHealthCheck(
        mqtt_client=_FakeMQTT(),
        twin_uuids=["twin-a"],
        stale_timeout=stale_timeout,
    )


def _stream_entry(health: EdgeHealthCheck) -> Dict[str, Any]:
    return health.get_health_data()["streams"]["stream"]


# --------------------------------------------------------------------------
# The policy constant
# --------------------------------------------------------------------------


def test_stale_timeout_is_the_documented_bound() -> None:
    """Freezing is unbounded, so staleness is the only bound on the lie.

    A cross-component contract, not a local knob: the frontend and the Go2 ROS
    bridges use 30, and ``cyberwave-cpp/.../camera_streaming.cpp`` keeps its
    own copy. Move them together.
    """
    assert SDK_EDGE_HEALTH_STALE_TIMEOUT_SECONDS == 30


def test_video_streamers_apply_the_policy_not_the_library_default() -> None:
    """``EdgeHealthCheck``'s own default stays 60 for unaudited callers.

    Media streamers produce continuously and take the tighter bound; drivers
    with device-dependent cadence (``BaseDriver._touch_edge_health``) keep the
    lenient default. Collapsing the two should be deliberate.
    """
    import inspect

    default = (
        inspect.signature(EdgeHealthCheck.__init__).parameters["stale_timeout"].default
    )
    assert default == 60
    assert SDK_EDGE_HEALTH_STALE_TIMEOUT_SECONDS < default


# --------------------------------------------------------------------------
# The counter-selection protocol
# --------------------------------------------------------------------------


def test_liveness_counter_prefers_captured_frame_count() -> None:
    track = _FakeTrack()
    track.emit(100)
    track.capture(3)

    assert track.frame_count == 103
    assert read_liveness_counter(track) == 3


def test_liveness_counter_falls_back_for_tracks_without_the_protocol() -> None:
    """Tracks with no degraded path keep the original ``frame_count`` behaviour.

    Opt-in by attribute presence is what makes this safe to land without
    auditing every track.
    """
    track = _FakeTrack(captured=None)
    track.emit(7)

    assert not hasattr(track, "captured_frame_count")
    assert read_liveness_counter(track) == 7


# --------------------------------------------------------------------------
# RealSense: same cached-frame fallback, same bug
# --------------------------------------------------------------------------


def _realsense_track(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Build a RealSenseVideoTrack without a device or pyrealsense2."""
    import numpy as np

    from cyberwave.sensor import camera_rs

    stub = MagicMock()

    def fake_from_ndarray(arr, format):  # noqa: A002 - mirrors the av API
        del arr, format
        vf = MagicMock()
        vf.pts = 0
        vf.reformat.return_value = vf
        return vf

    stub.from_ndarray.side_effect = fake_from_ndarray
    monkeypatch.setattr(camera_rs, "VideoFrame", stub)

    track = camera_rs.RealSenseVideoTrack.__new__(camera_rs.RealSenseVideoTrack)
    track.__class__.__bases__[0].__init__(track)
    track.frame_count = 0
    track.frame_0_timestamp = 0
    track.frame_0_timestamp_monotonic = 0
    track.color_fps = 30
    track.time_reference = None
    track.frame_callback = None
    track.depth_callback = None
    track.enable_depth = False
    track.depth_publish_interval = 1
    track.client = None
    track.twin_uuid = None
    track._cached_color_image = np.zeros((8, 8, 3), dtype=np.uint8)
    track._cached_depth_image = None
    track._capture_timestamp = lambda _ref: (0.0, 0.0)
    track._store_frame_metadata_for_sync = lambda **_kwargs: None
    track._capture_sync_frame = lambda *_args, **_kwargs: None
    return track


def test_realsense_cached_frames_do_not_count_as_captures(monkeypatch) -> None:
    """``camera_rs`` has the same fallback and needs the same treatment.

    Otherwise the depth-camera half of the driver keeps reporting a
    disconnected RealSense as healthy.
    """
    import numpy as np

    track = _realsense_track(monkeypatch)
    track._get_frames = lambda: (
        True,
        (np.zeros((8, 8, 3), dtype=np.uint8), None),
    )

    asyncio.run(track.recv())
    assert (track.frame_count, track.captured_frame_count) == (1, 1)

    # Device disconnected: _get_frames reports failure, cache carries the tick.
    track._get_frames = lambda: (False, None)
    for _ in range(4):
        assert asyncio.run(track.recv()) is not None

    assert track.frame_count == 5
    assert track.captured_frame_count == 1


def test_realsense_degraded_path_is_paced(monkeypatch) -> None:
    """A disconnected RealSense raises immediately; don't free-run on it."""
    from cyberwave.sensor import camera_rs

    track = _realsense_track(monkeypatch)
    track._get_frames = lambda: (False, None)

    real_sleep = asyncio.sleep
    seen: list[float] = []

    async def _recording(seconds):
        seen.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(camera_rs.asyncio, "sleep", _recording)

    asyncio.run(track.recv())

    assert seen == [pytest.approx(1 / 30)]


# --------------------------------------------------------------------------
# Tracks whose degraded path emits a *synthetic* frame rather than a freeze
#
# These have no device to lose, so it's tempting to call them incapable of
# emitting without capturing. They aren't: each substitutes something for a
# provider that returned nothing, which keeps ``frame_count`` climbing exactly
# like the freeze fallback does.
# --------------------------------------------------------------------------


def test_virtual_track_placeholder_does_not_count_as_capture() -> None:
    """``VirtualVideoTrack`` swaps in a placeholder when ``get_frame`` is dry."""
    import numpy as np

    from cyberwave.sensor.camera_virtual import VirtualVideoTrack

    supply: List[Optional[Any]] = [np.zeros((8, 8, 3), dtype=np.uint8)]
    track = VirtualVideoTrack(lambda: supply[0], width=8, height=8, fps=240)

    asyncio.run(track.recv())
    assert (track.frame_count, track.captured_frame_count) == (1, 1)

    # Provider dries up; the placeholder keeps the encoder fed.
    supply[0] = None
    for _ in range(4):
        assert asyncio.run(track.recv()) is not None

    assert track.frame_count == 5
    assert track.captured_frame_count == 1
    assert read_liveness_counter(track) == 1


def test_callback_track_placeholder_does_not_count_as_capture() -> None:
    """Same shape in ``CallbackVideoTrack``, including the bad-shape fallback."""
    import numpy as np

    from cyberwave.sensor.camera_callback import CallbackVideoTrack

    supply: List[Optional[Any]] = [np.zeros((8, 8, 3), dtype=np.uint8)]
    track = CallbackVideoTrack(lambda: supply[0], width=8, height=8, fps=240)

    asyncio.run(track.recv())
    assert (track.frame_count, track.captured_frame_count) == (1, 1)

    supply[0] = None
    asyncio.run(track.recv())
    # A frame of the wrong shape is no more evidence of liveness than None.
    supply[0] = np.zeros((8, 8), dtype=np.uint8)
    asyncio.run(track.recv())

    assert track.frame_count == 3
    assert track.captured_frame_count == 1


def test_sim_track_placeholder_does_not_count_as_capture() -> None:
    """A stalled sim render thread must not read as a healthy stream."""
    import numpy as np

    # ``camera_sim`` does ``import mujoco`` at module scope, and mujoco is in
    # neither the ``camera`` nor the ``speaker`` extra that CI installs. Local
    # to this test rather than module-level: the other eight tests here have no
    # sim dependency and must keep running without it.
    pytest.importorskip("mujoco", reason="mujoco not installed")

    from cyberwave.sensor.camera_sim import SimVideoTrack, ThreadSafeFrameBuffer

    buffer = ThreadSafeFrameBuffer(fps=10_000)
    track = SimVideoTrack(buffer, width=8, height=8, fps=240)

    assert buffer.add_frame(np.zeros((8, 8, 3), dtype=np.uint8))
    asyncio.run(track.recv())
    assert (track.frame_count, track.captured_frame_count) == (1, 1)

    # Render thread stops. The buffer is NOT emptied — get_latest_frame keeps
    # handing back the same still, which is why presence is not evidence.
    for _ in range(3):
        assert asyncio.run(track.recv()) is not None
        assert buffer.get_latest_frame() is not None

    assert track.frame_count == 4
    assert track.captured_frame_count == 1

    # Sim comes back.
    assert buffer.add_frame(np.zeros((8, 8, 3), dtype=np.uint8))
    asyncio.run(track.recv())
    assert track.captured_frame_count == 2


def test_h264_passthrough_empty_packet_does_not_count_as_capture() -> None:
    """A stalled upstream encoder must not read as healthy.

    ``H264PacketVideoTrack`` emits an empty ``av.Packet`` when ``get_packet``
    has nothing — zero RTP payloads, but a ``frame_count`` tick all the same.
    """
    from cyberwave.sensor.camera_h264 import H264PacketVideoTrack

    access_unit = b"\x00\x00\x00\x01\x67\x42"
    supply: List[Optional[Any]] = [(access_unit, True)]
    track = H264PacketVideoTrack(lambda: supply[0], width=8, height=8, fps=240)

    asyncio.run(track.recv())
    assert (track.frame_count, track.captured_frame_count) == (1, 1)

    # Upstream encoder stalls: None, then a raising provider.
    supply[0] = None
    asyncio.run(track.recv())

    def _boom() -> Any:
        raise RuntimeError("gstreamer pipeline died")

    track.get_packet = _boom
    asyncio.run(track.recv())

    assert track.frame_count == 3
    assert track.captured_frame_count == 1
    assert read_liveness_counter(track) == 1


# --------------------------------------------------------------------------
# Startup: connecting, not connected
# --------------------------------------------------------------------------


def _declared_health() -> EdgeHealthCheck:
    """A publisher that has declared a stream but produced no data yet.

    Registering a ``stream_config`` is what puts a row on the wire before the
    first frame; without one the publisher emits an empty ``streams`` map and
    the dashboard shows nothing at all.
    """
    health = _health()
    health.register_stream_config(
        "stream",
        {"kind": "camera", "fps": 30, "resolution": "640x480", "source": "0"},
    )
    return health


def test_declared_stream_reads_connecting_until_first_data() -> None:
    """``last_frame_time`` is seeded at ``__init__``, so zero frames must not
    read as ``connected`` — that claims frames are flowing before any have.
    """
    health = _declared_health()

    entry = _stream_entry(health)
    assert entry["connection_state"] == "connecting"
    assert entry["ice_connection_state"] == "new"
    # Starting up is not an outage: consumers only treat disconnected/failed/
    # closed as down, so this must stay off the red path.
    assert entry["is_stale"] is False

    health.update_frame_count()
    assert _stream_entry(health)["connection_state"] == "connected"


def test_mark_alive_publishers_reach_connected_without_frame_count() -> None:
    """The trap: ``mark_alive`` keeps ``frame_count`` at zero on purpose.

    Keying ``connecting`` off ``frame_count > 0`` would strand every audio /
    IMU publisher in ``connecting`` forever.
    """
    health = _declared_health()
    health.mark_alive()

    entry = _stream_entry(health)
    assert health.frame_count == 0
    assert entry["connection_state"] == "connected"


def test_connecting_still_goes_stale_if_data_never_arrives() -> None:
    """Connecting is a startup grace, not an exemption from the timeout."""
    health = _declared_health()
    health.last_frame_time = time.time() - (SDK_EDGE_HEALTH_STALE_TIMEOUT_SECONDS + 1)

    entry = _stream_entry(health)
    assert entry["is_stale"] is True
    assert entry["connection_state"] == "disconnected"


def test_publisher_with_nothing_declared_emits_no_stream_row() -> None:
    """The bootstrap gap stays grey, not yellow — no row rather than a phantom."""
    assert _health().get_health_data()["streams"] == {}


# --------------------------------------------------------------------------
# The monitor loop
# --------------------------------------------------------------------------


def _drive_monitor(streamer: BaseVideoStreamer, ticks: int = 3) -> None:
    """Run the real ``_monitor_frame_count`` for a bounded number of passes.

    Cancels rather than flipping ``_is_running``: the loop only re-reads its
    condition after a 0.1 s sleep.
    """

    async def scenario() -> None:
        task = asyncio.ensure_future(streamer._monitor_frame_count())
        for _ in range(ticks):
            await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())


def _streamer_with(track: _FakeTrack, health: EdgeHealthCheck) -> CV2CameraStreamer:
    streamer = CV2CameraStreamer(
        client=_FakeMQTT(),
        camera_id=0,
        fps=30,
        resolution=Resolution.VGA,
        twin_uuid="twin-a",
    )
    streamer.streamer = track  # type: ignore[assignment]
    streamer._health_check = health
    streamer._last_frame_count = 0
    streamer._is_running = True
    return streamer


def test_freeze_frames_do_not_refresh_liveness() -> None:
    """The bug, end to end: emitted-but-not-captured frames must not count."""
    track = _FakeTrack()
    health = _health()
    streamer = _streamer_with(track, health)

    track.capture(1)
    _drive_monitor(streamer)
    assert health.frame_count == 1

    # Camera unplugged: the freeze fallback keeps the encoder fed.
    track.emit(500)
    _drive_monitor(streamer)

    assert health.frame_count == 1, "freeze frames must not read as liveness"


def test_unplugged_camera_goes_stale_on_schedule() -> None:
    """No captures for ``stale_timeout`` means red.

    Backdates ``last_frame_time`` rather than sleeping.
    """
    track = _FakeTrack()
    health = _health()
    streamer = _streamer_with(track, health)

    track.capture(1)
    _drive_monitor(streamer)
    assert _stream_entry(health)["is_healthy"] is True

    # Past the bound, with nothing but freeze frames.
    track.emit(1800)
    health.last_frame_time = time.time() - (SDK_EDGE_HEALTH_STALE_TIMEOUT_SECONDS + 1)
    _drive_monitor(streamer)

    entry = _stream_entry(health)
    assert entry["is_stale"] is True
    assert entry["is_healthy"] is False
    assert entry["connection_state"] == "disconnected"
    assert health.get_health_data()["healthy_streams"] == 0


def test_multimedia_streamer_monitor_agrees_with_the_base_one() -> None:
    """``MultimediaStreamer`` keeps its own copy of the monitor loop.

    Not a ``BaseVideoStreamer`` subclass, so the two can drift. A camera that
    stops capturing must read as stale under either.
    """
    from cyberwave.sensor.av_streamer import MultimediaStreamer

    track = _FakeTrack()
    health = _health()
    streamer = MultimediaStreamer.__new__(MultimediaStreamer)
    streamer.video_track = track
    streamer._health_check = health
    streamer._last_frame_count = 0
    streamer._is_running = True
    streamer.pc = None

    track.capture(1)
    _drive_monitor(streamer)
    assert health.frame_count == 1

    track.emit(500)
    _drive_monitor(streamer)
    assert health.frame_count == 1


def test_blip_shorter_than_stale_timeout_stays_healthy() -> None:
    """The freeze fallback's original purpose survives the fix.

    An ffmpeg restart or IP-camera flake still rides through green; only an
    outage longer than ``stale_timeout`` changes the reported state.
    """
    track = _FakeTrack()
    health = _health()
    streamer = _streamer_with(track, health)

    track.capture(1)
    _drive_monitor(streamer)

    # A blip comfortably inside the bound: freeze frames only.
    track.emit(300)
    health.last_frame_time = time.time() - (SDK_EDGE_HEALTH_STALE_TIMEOUT_SECONDS / 3)
    _drive_monitor(streamer)
    assert _stream_entry(health)["is_healthy"] is True

    # Source comes back.
    track.capture(5)
    _drive_monitor(streamer)

    entry = _stream_entry(health)
    assert entry["is_healthy"] is True
    assert entry["connection_state"] == "connected"
    assert health.frame_count == 6
