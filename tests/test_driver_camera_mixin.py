"""``CameraStreamMixin`` / ``BaseCameraDriver`` wiring with a fake client, bus and streamer."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from cyberwave.driver import (
    BaseCameraDriver,
    CameraConfigurationError,
    CameraHardwareError,
    CameraSourceConfig,
    VideoSyncRequiredError,
)
from cyberwave.driver.sensors import camera as camera_module
from cyberwave.driver.sensors import camera_source

TWIN_UUID = "00000000-0000-0000-0000-00000000ca11"


class FakeBus:
    key_prefix = "cw"

    def __init__(self) -> None:
        self._backend = self
        self.backend = self
        self.subscriptions: dict[str, Any] = {}
        self.published: list[tuple[str, Any]] = []

    def subscribe(self, channel: str, callback: Any, **_kw: Any) -> Any:
        self.subscriptions[channel] = callback
        return self

    def publish(self, channel: str, payload: Any, metadata: Any = None) -> None:
        self.published.append((channel, payload))

    def publish_raw(self, channel: str, payload: bytes) -> None:
        self.published.append((channel, payload))

    def close(self) -> None:
        pass


class FakeStreamer:
    auto_reconnect = True

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.started = False
        self.stopped = False
        self._should_record = True
        self._should_reconnect = False

    async def start(self) -> None:
        if self.fail:
            raise RuntimeError("device busy")
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def run_with_auto_reconnect(self, stop_event: asyncio.Event, subscribe_to_commands: bool) -> None:
        assert subscribe_to_commands is False
        await stop_event.wait()

    async def _handle_start_command(self, callback: Any) -> None:
        callback("ok", "started")

    async def _handle_stop_command(self, callback: Any) -> None:
        callback("error", "boom")


class FakeMqtt:
    topic_prefix = ""

    def __init__(self) -> None:
        self.published: list[tuple[str, Any]] = []
        self.subscribed: dict[str, Any] = {}
        self.subscriber_keys: dict[str, Any] = {}

    def publish(self, topic: str, payload: Any, **_kw: Any) -> None:
        self.published.append((topic, payload))

    def subscribe(self, topic: str, handler: Any, *, subscriber_key: Any = None) -> None:
        self.subscribed[topic] = handler
        self.subscriber_keys[topic] = subscriber_key


class FakeClient:
    def __init__(self, bus: FakeBus | None = None) -> None:
        self.mqtt = FakeMqtt()
        self.bus = bus
        self.streams: list[dict[str, Any]] = []

    def data_bus_for(self, _twin_uuid: str) -> FakeBus:
        assert self.bus is not None
        return self.bus

    def _build_video_streamer(self, **kwargs: Any) -> FakeStreamer:
        self.streams.append(kwargs)
        return FakeStreamer(fail=kwargs["camera_id"] == "broken")


class FakeTwin:
    uuid = TWIN_UUID
    capabilities = {"sensors": [{"id": "color_camera", "type": "rgb"}]}
    default_camera_name = "color_camera"


class Camera(BaseCameraDriver):
    pass


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(camera_source, "gstreamer_available", lambda: True)
    for name in (
        "CYBERWAVE_TWIN_JSON_FILE",
        "CYBERWAVE_DATA_BACKEND",
        "CYBERWAVE_METADATA_FRAME_FILTER_ENABLED",
        "CYBERWAVE_METADATA_VIDEO_DEVICE",
        "CYBERWAVE_METADATA_GST_SOURCE",
        "CYBERWAVE_DETECTION_OVERLAYS",
        "CYBERWAVE_SYNC_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)


def _driver(bus: FakeBus | None = None, cls: type[BaseCameraDriver] = Camera) -> BaseCameraDriver:
    driver = cls(twin=FakeTwin(), client=FakeClient(bus))
    driver.raise_alert = lambda alert: None  # type: ignore[method-assign]
    driver.resolve_alert = lambda *a: None  # type: ignore[method-assign]
    return driver


def _with_bus(monkeypatch: pytest.MonkeyPatch) -> FakeBus:
    monkeypatch.setenv("CYBERWAVE_DATA_BACKEND", "zenoh")
    monkeypatch.setenv("CYBERWAVE_PUBLISH_MODE", "dual")
    return FakeBus()


def test_streamer_publishes_edge_health_not_the_driver() -> None:
    assert Camera.EDGE_HEALTH_ENABLED is False
    assert Camera.auto_register_interface is False


def test_registry_id_comes_from_twin_json(tmp_path, monkeypatch) -> None:
    twin_json = tmp_path / "twin.json"
    twin_json.write_text(json.dumps({"asset": {"registry_id": "acme/cam"}}))
    monkeypatch.setenv("CYBERWAVE_TWIN_JSON_FILE", str(twin_json))

    assert _driver().registry_id == "acme/cam"


def test_sync_required_without_extension_refuses_to_start(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cyberwave_video_sync", None)

    class StrictCamera(BaseCameraDriver):
        VIDEO_SYNC_REQUIRED = True

    with pytest.raises(VideoSyncRequiredError):
        _driver(cls=StrictCamera).configure_cameras()


def test_sync_required_cannot_be_switched_off(monkeypatch) -> None:
    fake = type(sys)("cyberwave_video_sync")
    fake.install = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cyberwave_video_sync", fake)
    monkeypatch.setenv("CYBERWAVE_SYNC_ENABLED", "0")

    class StrictCamera(BaseCameraDriver):
        VIDEO_SYNC_REQUIRED = True

    with pytest.raises(VideoSyncRequiredError, match="CYBERWAVE_SYNC_ENABLED"):
        _driver(cls=StrictCamera).configure_cameras()


def test_frame_filter_without_bus_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("CYBERWAVE_METADATA_FRAME_FILTER_ENABLED", "true")
    driver = _driver()

    with pytest.raises(CameraConfigurationError, match="data bus"):
        asyncio.run(driver.start_cameras_async())


def test_frame_filter_with_pre_encoded_source_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("CYBERWAVE_METADATA_FRAME_FILTER_ENABLED", "true")
    monkeypatch.setenv("CYBERWAVE_METADATA_GST_SOURCE", "videotestsrc ! x264enc")

    with pytest.raises(CameraConfigurationError, match="pre-encoded"):
        _driver().configure_cameras()


def test_start_opens_stream_and_announces_it(monkeypatch) -> None:
    bus = _with_bus(monkeypatch)
    driver = _driver(bus)

    async def _run() -> None:
        await driver.start_cameras_async()
        await driver.stop_cameras_async()

    asyncio.run(_run())

    stream = driver._cw.streams[0]
    assert stream["camera_type"] == "cv2"
    assert stream["camera_name"] == "color_camera"
    assert stream["frame_callback"].func == driver._on_camera_frame
    assert f"cw/{TWIN_UUID}/data/detections/**" in bus.subscriptions
    assert "frames/overlay" in bus.subscriptions
    topics = [topic for topic, _ in driver._cw.mqtt.published]
    assert f"cyberwave/twin/{TWIN_UUID}/start_video/status" in topics


def test_pinned_camera_never_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("CYBERWAVE_METADATA_VIDEO_DEVICE", "rtsp://broken")
    driver = _driver()
    driver.CAMERA_SOURCE = CameraSourceConfig(camera_id="broken", pinned=True)
    monkeypatch.setattr(camera_module, "list_cameras", lambda: pytest.fail("must not scan"))

    with pytest.raises(CameraHardwareError, match="Not falling back"):
        asyncio.run(driver.start_cameras_async())


def test_unpinned_camera_falls_back_to_detected_device(monkeypatch) -> None:
    driver = _driver()
    driver.CAMERA_SOURCE = CameraSourceConfig(camera_id="broken")
    monkeypatch.setattr(camera_module, "list_cameras", lambda: (["3"], []))

    asyncio.run(driver.start_cameras_async())

    assert [s["camera_id"] for s in driver._cw.streams] == ["broken", 3]
    assert driver.camera_config.camera_id == 3


def test_no_camera_at_all_is_a_hardware_error(monkeypatch) -> None:
    driver = _driver()
    driver.CAMERA_SOURCE = CameraSourceConfig(camera_id="broken")
    monkeypatch.setattr(camera_module, "list_cameras", lambda: ([], []))

    with pytest.raises(CameraHardwareError, match="No camera hardware"):
        asyncio.run(driver.start_cameras_async())


def test_video_commands_delegate_to_the_streamer(monkeypatch) -> None:
    driver = _driver()

    async def _run() -> None:
        await driver.start_cameras_async()
        driver._video_command_payload = {"command": "start_video", "data": {"recording": False}}
        await driver._on_start_video_async()
        assert driver.camera_streamer._should_record is False
        with pytest.raises(RuntimeError, match="boom"):
            await driver._on_stop_video_async()

    asyncio.run(_run())


class TestFramePipeline:
    def _started(self, monkeypatch, cls: type[BaseCameraDriver] = Camera) -> BaseCameraDriver:
        bus = _with_bus(monkeypatch)
        driver = _driver(bus, cls=cls)
        asyncio.run(driver.start_cameras_async())
        return driver

    def test_untouched_frames_are_published_zero_copy(self, monkeypatch) -> None:
        driver = self._started(monkeypatch)
        frame = np.full((4, 4, 3), 7, dtype=np.uint8)

        driver._on_camera_frame(driver.cameras["color_camera"], frame, 0)

        assert driver._zenoh_frame_slot.take(timeout=0.1) is frame
        asyncio.run(driver.stop_cameras_async())

    def test_filter_blanks_the_encoded_frame_but_not_the_published_one(self, monkeypatch) -> None:
        monkeypatch.setenv("CYBERWAVE_METADATA_FRAME_FILTER_ENABLED", "true")
        driver = self._started(monkeypatch)
        frame = np.full((4, 4, 3), 7, dtype=np.uint8)

        driver._on_camera_frame(driver.cameras["color_camera"], frame, 0)

        assert not frame.any(), "stale filter must blank the encoded frame"
        published = driver._zenoh_frame_slot.take(timeout=0.1)
        assert published is not frame and (published == 7).all()
        asyncio.run(driver.stop_cameras_async())

    def test_filter_substitutes_a_fresh_worker_frame(self, monkeypatch) -> None:
        monkeypatch.setenv("CYBERWAVE_METADATA_FRAME_FILTER_ENABLED", "true")
        driver = self._started(monkeypatch)
        driver._zenoh_data_bus.subscriptions["frames/filtered"](
            np.full((4, 4, 3), 99, dtype=np.uint8)
        )
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        driver._on_camera_frame(driver.cameras["color_camera"], frame, 0)

        assert (frame == 99).all()
        asyncio.run(driver.stop_cameras_async())

    def test_custom_hook_edits_the_encoded_copy(self, monkeypatch) -> None:
        class Tinted(BaseCameraDriver):
            def on_camera_frame(self, frame: np.ndarray, camera: str) -> None:
                frame[:] = 1

        driver = self._started(monkeypatch, cls=Tinted)
        frame = np.full((4, 4, 3), 7, dtype=np.uint8)

        driver._on_camera_frame(driver.cameras["color_camera"], frame, 0)

        assert (frame == 1).all()
        assert (driver._zenoh_frame_slot.take(timeout=0.1) == 7).all()
        asyncio.run(driver.stop_cameras_async())


def test_pre_encoded_source_streams_through_gstreamer(monkeypatch) -> None:
    camera_gstreamer = pytest.importorskip("cyberwave.sensor.camera_gstreamer")

    built: dict[str, Any] = {}

    class _Recorder(FakeStreamer):
        def __init__(self, mqtt: Any, **kwargs: Any) -> None:
            super().__init__()
            built.update(kwargs, mqtt=mqtt)

    monkeypatch.setattr(camera_gstreamer, "GStreamerH264CameraStreamer", _Recorder)
    monkeypatch.setenv("CYBERWAVE_METADATA_GST_SOURCE", "v4l2src ! video/x-h264")
    driver = _driver()

    asyncio.run(driver.start_cameras_async())

    assert built["source"] == "v4l2src ! video/x-h264"
    assert built["camera_name"] == "color_camera"
    assert built["frame_callback"].func == driver._on_decoded_frame
    assert built["require_sync"] is False
    assert driver._cw.streams == [], "pre-encoded sources must not go through the cv2/RealSense factory"


def test_video_commands_from_any_view_are_honoured() -> None:
    driver = _driver()
    handled: list[str] = []

    async def _run() -> None:
        async def _record(payload: Any = None) -> None:
            handled.append("start")

        driver._handle_start_video_async = _record  # type: ignore[method-assign]
        await driver.register_video_commands_async()
        on_command = driver._cw.mqtt.subscribed[f"cyberwave/twin/{TWIN_UUID}/command"]
        on_command({"command": "start_video", "source_type": "edit"})
        await asyncio.sleep(0.05)

    asyncio.run(_run())

    assert handled == ["start"]


def test_pre_encoded_source_without_gstreamer_is_a_config_error(monkeypatch) -> None:
    monkeypatch.setattr(camera_source, "gstreamer_available", lambda: False)
    monkeypatch.setenv("CYBERWAVE_METADATA_GST_SOURCE", "appsrc")

    with pytest.raises(CameraConfigurationError, match="-gstreamer"):
        _driver().configure_cameras()


def test_push_h264_reaches_the_streamer_while_it_starts(monkeypatch) -> None:
    """A push-mode source sees its first keyframe only through push_h264()."""
    camera_gstreamer = pytest.importorskip("cyberwave.sensor.camera_gstreamer")

    pushed: list[tuple[bytes, float | None]] = []
    driver = _driver()

    class _PushStreamer(FakeStreamer):
        def __init__(self, mqtt: Any, **kwargs: Any) -> None:
            super().__init__()

        def push(self, data: bytes, *, capture_time: float | None = None) -> bool:
            pushed.append((data, capture_time))
            return True

        async def start(self) -> None:
            assert driver.push_h264(b"\x00\x00\x00\x01\x65", capture_time=12.5)
            await super().start()

    monkeypatch.setattr(camera_gstreamer, "GStreamerH264CameraStreamer", _PushStreamer)
    monkeypatch.setenv("CYBERWAVE_METADATA_GST_SOURCE", "appsrc")

    assert driver.push_h264(b"early") is False
    asyncio.run(driver.start_cameras_async())

    assert pushed == [(b"\x00\x00\x00\x01\x65", 12.5)]


# ── The mixin on a driver that is not a camera (e.g. a robot bridge) ─────────

from cyberwave.driver import BaseDriver, CameraStreamMixin  # noqa: E402
from cyberwave.driver.status import DriverLifecycleState  # noqa: E402


class RobotBridge(CameraStreamMixin, BaseDriver):
    REGISTRY_ID = "acme/robot"
    CAMERA_SOURCE = CameraSourceConfig(camera_id=0, camera_name="front_camera")
    CAMERA_START_RETRY_S = 0.01

    @classmethod
    def create(cls) -> RobotBridge:
        return cls()

    async def on_configure(self) -> None: ...
    async def on_connect_to_device(self) -> None: ...
    async def on_register_callbacks(self) -> None: ...
    async def on_activate(self) -> None: ...
    async def on_shutdown(self) -> None: ...


def _bridge(client: FakeClient | None = None) -> RobotBridge:
    bridge = RobotBridge(twin=FakeTwin(), client=client or FakeClient())
    bridge.raise_alert = MagicMock()  # type: ignore[method-assign]
    bridge.resolve_alert = lambda *a: None  # type: ignore[method-assign]
    return bridge


async def _monitor_briefly(bridge: RobotBridge, until: Any) -> None:
    task = asyncio.create_task(bridge.on_start_monitoring())
    for _ in range(200):
        if until():
            break
        await asyncio.sleep(0.01)
    bridge.request_shutdown()
    await asyncio.wait_for(task, timeout=5)


def test_robot_bridge_keeps_its_own_heartbeat_and_lifecycle() -> None:
    assert RobotBridge.EDGE_HEALTH_ENABLED is True
    assert RobotBridge.VIDEO_COMMANDS_DRIVE_LIFECYCLE is False
    assert BaseCameraDriver.EDGE_HEALTH_ENABLED is False
    assert BaseCameraDriver.VIDEO_COMMANDS_DRIVE_LIFECYCLE is True


def test_mixin_runs_itself_from_on_start_monitoring() -> None:
    bridge = _bridge()

    asyncio.run(_monitor_briefly(bridge, until=lambda: bridge.camera_streamer is not None))

    assert [s["camera_name"] for s in bridge._cw.streams] == ["front_camera"]
    command_topic = f"cyberwave/twin/{TWIN_UUID}/command"
    assert command_topic in bridge._cw.mqtt.subscribed
    # Not the default slot, which the driver's own command dispatch holds.
    assert bridge._cw.mqtt.subscriber_keys[command_topic] is not None
    assert bridge.camera_streamer is None, "stopped at shutdown"


def test_stop_video_does_not_mark_the_robot_inactive() -> None:
    bridge = _bridge()
    bridge._lifecycle_state = DriverLifecycleState.ACTIVE

    async def _run() -> None:
        await bridge.start_cameras_async()
        bridge._streaming_active = True
        bridge.camera_streamer._handle_stop_command = AsyncMock(
            side_effect=lambda cb: cb("ok", "stopped")
        )
        await bridge._handle_stop_video_async()

    asyncio.run(_run())

    assert bridge._lifecycle_state == DriverLifecycleState.ACTIVE


def test_a_failing_camera_retries_instead_of_ending_the_driver(monkeypatch) -> None:
    monkeypatch.setattr(camera_module, "list_cameras", lambda: ([], []))
    client = FakeClient()
    attempts = {"n": 0}
    real_build = client._build_video_streamer

    def _flaky(**kwargs: Any) -> FakeStreamer:
        attempts["n"] += 1
        return FakeStreamer(fail=attempts["n"] < 3) if attempts["n"] < 3 else real_build(**kwargs)

    client._build_video_streamer = _flaky  # type: ignore[method-assign]
    bridge = _bridge(client)

    asyncio.run(_monitor_briefly(bridge, until=lambda: bridge.camera_streamer is not None))

    assert attempts["n"] == 3
    assert bridge.raise_alert.call_count == 2, "one camera alert per failed attempt"


def test_base_camera_driver_does_not_start_twice() -> None:
    driver = _driver()

    async def _run() -> None:
        await driver.start_cameras_async()
        task = asyncio.create_task(driver.on_start_monitoring())
        await asyncio.sleep(0.05)
        driver.request_shutdown()
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(_run())

    assert len(driver._cw.streams) == 1
