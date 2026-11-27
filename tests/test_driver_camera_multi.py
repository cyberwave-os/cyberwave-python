"""Several cameras in one driver: each streams, fails and recovers on its own."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from cyberwave.driver import (
    BaseCameraDriver,
    BaseDriver,
    CameraConfigurationError,
    CameraHardwareError,
    CameraSourceConfig,
    CameraStatus,
    CameraStreamMixin,
)
from cyberwave.driver.sensors import camera as camera_module
from tests.test_driver_camera_mixin import (  # noqa: F401  (_env is an autouse fixture)
    TWIN_UUID,
    FakeBus,
    FakeClient,
    FakeStreamer,
    FakeTwin,
    _env,
    _with_bus,
)

WRIST = CameraSourceConfig(camera_id="/dev/video0", camera_name="wrist_camera")
BASE = CameraSourceConfig(camera_id="/dev/video2", camera_name="base_camera")
BROKEN = CameraSourceConfig(camera_id="broken", camera_name="overhead_camera")


class ArmDriver(CameraStreamMixin, BaseDriver):
    REGISTRY_ID = "acme/arm"
    CAMERAS = [WRIST, BASE, BROKEN]
    CAMERA_START_RETRY_S = 0.01

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.statuses: list[tuple[str, CameraStatus]] = []
        self.raise_alert = MagicMock()  # type: ignore[method-assign]
        self.resolve_alert = MagicMock()  # type: ignore[method-assign]

    @classmethod
    def create(cls) -> ArmDriver:
        return cls()

    def on_camera_status(self, camera: str, status: CameraStatus, reason: str | None) -> None:
        self.statuses.append((camera, status))

    async def on_configure(self) -> None: ...
    async def on_connect_to_device(self) -> None: ...
    async def on_register_callbacks(self) -> None: ...
    async def on_activate(self) -> None: ...
    async def on_shutdown(self) -> None: ...


def _arm(bus: FakeBus | None = None, cls: type[ArmDriver] = ArmDriver) -> ArmDriver:
    return cls(twin=FakeTwin(), client=FakeClient(bus))


async def _monitor(driver: ArmDriver, until: Any) -> None:
    task = asyncio.create_task(driver.on_start_monitoring())
    for _ in range(300):
        if until():
            break
        await asyncio.sleep(0.01)
    driver.request_shutdown()
    await asyncio.wait_for(task, timeout=5)


def test_a_failing_camera_leaves_the_others_and_the_driver_running() -> None:
    driver = _arm()
    retries = lambda: driver.statuses.count(("overhead_camera", CameraStatus.RETRYING))  # noqa: E731

    asyncio.run(_monitor(driver, until=lambda: retries() >= 2))

    opened = [s["camera_name"] for s in driver._cw.streams]
    assert opened.count("wrist_camera") == 1 and opened.count("base_camera") == 1
    assert opened.count("overhead_camera") >= 2, "the broken camera is retried"
    assert ("wrist_camera", CameraStatus.STREAMING) in driver.statuses
    assert ("base_camera", CameraStatus.STREAMING) in driver.statuses
    assert ("overhead_camera", CameraStatus.STREAMING) not in driver.statuses
    # Each camera has its own alert; only the broken one raised.
    components = {call.args[0].component for call in driver.raise_alert.call_args_list}
    assert components == {"camera/overhead_camera"}
    assert all(s.status is CameraStatus.STOPPED for s in driver.cameras.values())


def test_cameras_are_listed_in_declaration_order() -> None:
    driver = _arm()
    driver.configure_cameras()

    assert list(driver.cameras) == ["wrist_camera", "base_camera", "overhead_camera"]
    assert driver.cameras["wrist_camera"].primary
    assert driver.camera_config is WRIST


def test_each_camera_publishes_its_own_frames_channel(monkeypatch) -> None:
    bus = _with_bus(monkeypatch)

    class TwoCameras(ArmDriver):
        CAMERAS = [WRIST, BASE]

    driver = _arm(bus, cls=TwoCameras)
    asyncio.run(driver.start_cameras_async())
    try:
        wrist, base = np.full((2, 2, 3), 1, np.uint8), np.full((2, 2, 3), 2, np.uint8)
        driver._on_camera_frame(driver.cameras["wrist_camera"], wrist, 0)
        driver._on_camera_frame(driver.cameras["base_camera"], base, 0)

        assert driver._zenoh_frame_slot.take(timeout=0.5) is wrist
        deadline = time.monotonic() + 2
        while not any(ch == "frames/base_camera" for ch, _ in bus.published):
            assert time.monotonic() < deadline, bus.published
            time.sleep(0.01)
    finally:
        asyncio.run(driver.stop_cameras_async())
    assert [s["camera_name"] for s in driver._cw.streams] == ["wrist_camera", "base_camera"]


def _detections(sensor: str | None = None) -> bytes:
    payload: dict[str, Any] = {
        "detections": [{"label": "cup", "confidence": 0.9, "x1": 0, "y1": 0, "x2": 3, "y2": 3}],
        "frame_width": 4,
        "frame_height": 4,
    }
    if sensor is not None:
        payload["sensor"] = sensor
    return json.dumps(payload).encode()


def _started(monkeypatch, cameras: list[CameraSourceConfig]) -> tuple[ArmDriver, Any]:
    bus = _with_bus(monkeypatch)

    class Arm(ArmDriver):
        CAMERAS = cameras

    driver = _arm(bus, cls=Arm)
    asyncio.run(driver.start_cameras_async())
    return driver, bus.subscriptions[f"cw/{TWIN_UUID}/data/detections/**"]


def test_detections_are_drawn_on_the_camera_the_payload_names(monkeypatch) -> None:
    driver, on_detections = _started(monkeypatch, [WRIST, BASE])
    try:
        # Models publish on detections/<runtime>; the camera is in the payload.
        on_detections(MagicMock(channel="detections/onnxruntime", payload=_detections("base_camera")))

        assert driver.cameras["base_camera"]._detection_cache.snapshot() is not None
        assert driver.cameras["wrist_camera"]._detection_cache.snapshot() is None
    finally:
        asyncio.run(driver.stop_cameras_async())


def test_unrouted_detections_are_not_drawn_on_a_guessed_camera(monkeypatch) -> None:
    driver, on_detections = _started(monkeypatch, [WRIST, BASE])
    try:
        on_detections(MagicMock(channel="detections/onnxruntime", payload=_detections()))

        assert all(c._detection_cache.snapshot() is None for c in driver.cameras.values())
    finally:
        asyncio.run(driver.stop_cameras_async())


def test_a_single_camera_draws_detections_that_name_no_sensor(monkeypatch) -> None:
    driver, on_detections = _started(monkeypatch, [WRIST])
    try:
        on_detections(MagicMock(channel="detections/onnxruntime", payload=_detections()))

        assert driver.cameras["wrist_camera"]._detection_cache.snapshot() is not None
    finally:
        asyncio.run(driver.stop_cameras_async())


def test_video_commands_target_the_named_sensor_or_every_camera() -> None:
    class TwoCameras(ArmDriver):
        CAMERAS = [WRIST, BASE]

    driver = _arm(cls=TwoCameras)
    stopped: list[str] = []

    async def _run() -> None:
        await driver.start_cameras_async()
        for name, stream in driver.cameras.items():

            async def _stop(cb: Any, name: str = name) -> None:
                stopped.append(name)
                cb("ok", "stopped")

            stream.streamer._handle_stop_command = _stop
        driver._video_command_payload = {"command": "stop_video", "data": {"sensor": "base_camera"}}
        await driver._on_stop_video_async()
        assert stopped == ["base_camera"]
        driver._video_command_payload = {"command": "stop_video"}
        await driver._on_stop_video_async()
        await driver.stop_cameras_async()

    asyncio.run(_run())

    assert stopped == ["base_camera", "wrist_camera", "base_camera"]


def test_push_h264_routes_by_camera_name(monkeypatch) -> None:
    camera_gstreamer = pytest.importorskip("cyberwave.sensor.camera_gstreamer")

    pushed: list[tuple[str, bytes]] = []

    class _PushStreamer(FakeStreamer):
        def __init__(self, mqtt: Any, **kwargs: Any) -> None:
            super().__init__()
            self.name = kwargs["camera_name"]

        def push(self, data: bytes, *, capture_time: float | None = None) -> bool:
            pushed.append((self.name, data))
            return True

    monkeypatch.setattr(camera_gstreamer, "GStreamerH264CameraStreamer", _PushStreamer)

    class PushArm(ArmDriver):
        CAMERAS = [
            CameraSourceConfig.push(camera_name="left_eye"),
            CameraSourceConfig.push(camera_name="right_eye"),
        ]

    driver = _arm(cls=PushArm)
    asyncio.run(driver.start_cameras_async())

    assert driver.push_h264(b"L", camera="left_eye")
    assert driver.push_h264(b"R", camera="right_eye")
    with pytest.raises(ValueError, match="several push cameras"):
        driver.push_h264(b"?")
    with pytest.raises(ValueError, match="no camera named"):
        driver.push_h264(b"?", camera="nose")
    assert pushed == [("left_eye", b"L"), ("right_eye", b"R")]


def test_on_camera_frame_names_the_camera(monkeypatch) -> None:
    seen: list[str] = []

    class Tagged(ArmDriver):
        CAMERAS = [WRIST, BASE]

        def on_camera_frame(self, frame: np.ndarray, camera: str) -> None:
            seen.append(camera)

    driver = _arm(cls=Tagged)
    driver.configure_cameras()
    frame = np.zeros((2, 2, 3), np.uint8)
    driver._on_camera_frame(driver.cameras["base_camera"], frame, 0)

    assert seen == ["base_camera"]


class TestConfiguration:
    def test_each_camera_needs_a_name(self) -> None:
        class Unnamed(ArmDriver):
            CAMERAS = [WRIST, CameraSourceConfig(camera_id=3)]

        with pytest.raises(CameraConfigurationError, match="camera_name"):
            _arm(cls=Unnamed).configure_cameras()

    def test_names_are_distinct(self) -> None:
        class Twice(ArmDriver):
            CAMERAS = [WRIST, CameraSourceConfig(camera_id=3, camera_name="wrist_camera")]

        with pytest.raises(CameraConfigurationError, match="repeated: wrist_camera"):
            _arm(cls=Twice).configure_cameras()

    def test_the_sensorless_frame_filter_refuses_several_cameras(self, monkeypatch) -> None:
        monkeypatch.setenv("CYBERWAVE_METADATA_FRAME_FILTER_ENABLED", "true")

        with pytest.raises(CameraConfigurationError, match="unfiltered"):
            _arm().configure_cameras()

    def test_cameras_never_fall_back_to_another_device(self, monkeypatch) -> None:
        monkeypatch.setattr(camera_module, "list_cameras", lambda: pytest.fail("must not scan"))

        class TwoCameras(ArmDriver):
            CAMERAS = [WRIST, BROKEN]

        driver = _arm(cls=TwoCameras)
        asyncio.run(driver.start_cameras_async())

        assert driver.cameras["overhead_camera"].status is CameraStatus.RETRYING


class MultiCameraDriver(BaseCameraDriver):
    CAMERAS = [WRIST, BROKEN]


def _camera_driver(cls: type[BaseCameraDriver]) -> BaseCameraDriver:
    driver = cls(twin=FakeTwin(), client=FakeClient())
    driver.raise_alert = lambda alert: None  # type: ignore[method-assign]
    driver.resolve_alert = lambda *a: None  # type: ignore[method-assign]
    return driver


def test_camera_driver_activates_while_one_camera_is_down() -> None:
    driver = _camera_driver(MultiCameraDriver)

    asyncio.run(driver.on_activate())

    assert driver.cameras["wrist_camera"].status is CameraStatus.STREAMING
    assert driver.cameras["overhead_camera"].status is CameraStatus.RETRYING


def test_camera_driver_fails_when_no_camera_opens() -> None:
    class AllBroken(BaseCameraDriver):
        CAMERAS = [BROKEN, CameraSourceConfig(camera_id="broken", camera_name="rear_camera")]

    driver = _camera_driver(AllBroken)

    with pytest.raises(CameraHardwareError):
        asyncio.run(driver.on_activate())

    assert {s.status for s in driver.cameras.values()} == {CameraStatus.FAILED}


def test_a_usb_camera_by_serial_opens_its_by_id_node(tmp_path, monkeypatch) -> None:
    from cyberwave.driver.sensors import camera_source

    monkeypatch.setattr(camera_source, "V4L_BY_ID_DIR", str(tmp_path))

    class SerialArm(ArmDriver):
        CAMERAS = [
            CameraSourceConfig.usb(serial_number="0123ABCD", camera_name="wrist_camera"),
            BASE,
        ]

    driver = _arm(cls=SerialArm)
    asyncio.run(driver.start_cameras_async())
    # Not attached: that camera retries, the other one streams.
    assert driver.cameras["wrist_camera"].status is CameraStatus.RETRYING
    assert driver.cameras["base_camera"].status is CameraStatus.STREAMING

    node = tmp_path / "usb-Acme_Cam_0123ABCD-video-index0"
    node.touch()
    asyncio.run(driver._start_camera(driver.cameras["wrist_camera"]))

    assert driver._cw.streams[-1]["camera_id"] == str(node)
    assert driver.cameras["wrist_camera"].status is CameraStatus.STREAMING


# ── Review fixes: failed starts, credentials, command routing, FAILED ────────


def test_a_failed_start_shuts_its_streamer_down() -> None:
    """A source left running by a failed start would add one connection per retry."""
    built: list[FakeStreamer] = []

    class Broken(ArmDriver):
        CAMERAS = [WRIST, BROKEN]

    driver = _arm(cls=Broken)
    real_build = driver._cw._build_video_streamer

    def _build(**kwargs: Any) -> FakeStreamer:
        streamer = real_build(**kwargs)
        built.append(streamer)
        return streamer

    driver._cw._build_video_streamer = _build  # type: ignore[method-assign]
    asyncio.run(driver.start_cameras_async())

    failed = [s for s in built if s.fail]
    assert failed and all(s.stopped for s in failed)
    assert not any(s.stopped for s in built if not s.fail), "the camera that started keeps streaming"


def test_url_credentials_stay_out_of_errors_logs_and_alerts(caplog) -> None:
    url = "rtsp://admin:p@ss/w0rd@cam.lan:554/s"

    class IpCamera(ArmDriver):
        CAMERAS = [WRIST, CameraSourceConfig.from_url(url, "gate_camera", decode=True)]
        CAMERA_START_RETRY_S = 0.01

    driver = _arm(cls=IpCamera)
    real_build = driver._cw._build_video_streamer
    driver._cw._build_video_streamer = lambda **kw: (  # type: ignore[method-assign]
        FakeStreamer(fail=True) if kw["camera_name"] == "gate_camera" else real_build(**kw)
    )
    retried = lambda: driver.statuses.count(("gate_camera", CameraStatus.RETRYING)) >= 2  # noqa: E731

    with caplog.at_level("DEBUG"):
        asyncio.run(_monitor(driver, until=retried))

    alerts = [call.args[0] for call in driver.raise_alert.call_args_list]
    assert alerts, "the failing camera raised its alert"
    leaked = [
        text
        for text in [caplog.text, *(repr(a.to_dict()) for a in alerts)]
        if "p@ss" in text or "w0rd" in text
    ]
    assert not leaked


def test_close_video_commands_keep_their_own_camera() -> None:
    """Two commands queued before either runs must each act on the camera they name."""

    class TwoCameras(ArmDriver):
        CAMERAS = [WRIST, BASE]

    driver = _arm(cls=TwoCameras)
    stopped: list[str] = []

    async def _run() -> None:
        await driver.start_cameras_async()
        for name, stream in driver.cameras.items():

            async def _stop(cb: Any, name: str = name) -> None:
                stopped.append(name)
                cb("ok", "stopped")

            stream.streamer._handle_stop_command = _stop
        await driver.register_video_commands_async()
        on_command = driver._cw.mqtt.subscribed[f"cyberwave/twin/{TWIN_UUID}/command"]
        # Both arrive before the loop runs either handler.
        on_command({"command": "stop_video", "data": {"sensor": "wrist_camera"}})
        on_command({"command": "stop_video", "data": {"sensor": "base_camera"}})
        for _ in range(50):
            if len(stopped) == 2:
                break
            await asyncio.sleep(0.01)
        await driver.stop_cameras_async()

    asyncio.run(_run())

    assert sorted(stopped) == ["base_camera", "wrist_camera"]


def test_missing_frame_sync_fails_the_camera_instead_of_retrying(monkeypatch) -> None:
    camera_gstreamer = pytest.importorskip("cyberwave.sensor.camera_gstreamer")
    starts: list[str] = []

    class NoSyncStreamer(FakeStreamer):
        def __init__(self, mqtt: Any, **kwargs: Any) -> None:
            super().__init__()

        async def start(self) -> None:
            starts.append("start")
            raise camera_gstreamer.VideoSyncUnavailableError("probe not installed")

    monkeypatch.setattr(camera_gstreamer, "GStreamerH264CameraStreamer", NoSyncStreamer)
    monkeypatch.setattr(ArmDriver, "_check_video_sync", lambda self, config: None)

    class PushArm(ArmDriver):
        CAMERAS = [WRIST, CameraSourceConfig.push(camera_name="nose_camera")]

    driver = _arm(cls=PushArm)
    failed = lambda: ("nose_camera", CameraStatus.FAILED) in driver.statuses  # noqa: E731

    asyncio.run(_monitor(driver, until=failed))

    assert failed()
    assert ("nose_camera", CameraStatus.RETRYING) not in driver.statuses
    assert starts == ["start"], "not retried"
    assert ("wrist_camera", CameraStatus.STREAMING) in driver.statuses


def test_a_single_camera_draws_tracked_detections_named_default(monkeypatch) -> None:
    """Tracking without stream= publishes sensor "default": the one camera still draws it."""
    driver, on_detections = _started(monkeypatch, [WRIST])
    try:
        on_detections(MagicMock(channel="detections/onnxruntime", payload=_detections("default")))

        assert driver.cameras["wrist_camera"]._detection_cache.snapshot() is not None
    finally:
        asyncio.run(driver.stop_cameras_async())


def test_several_cameras_drop_detections_for_an_unknown_sensor(monkeypatch) -> None:
    driver, on_detections = _started(monkeypatch, [WRIST, BASE])
    try:
        on_detections(MagicMock(channel="detections/onnxruntime", payload=_detections("default")))

        assert all(c._detection_cache.snapshot() is None for c in driver.cameras.values())
    finally:
        asyncio.run(driver.stop_cameras_async())


def test_a_pre_encoded_second_camera_makes_unnamed_detections_ambiguous(monkeypatch) -> None:
    camera_gstreamer = pytest.importorskip("cyberwave.sensor.camera_gstreamer")

    class _Streamer(FakeStreamer):
        def __init__(self, mqtt: Any, **kwargs: Any) -> None:
            super().__init__()

    monkeypatch.setattr(camera_gstreamer, "GStreamerH264CameraStreamer", _Streamer)
    monkeypatch.setattr(ArmDriver, "_check_video_sync", lambda self, config: None)
    driver, on_detections = _started(
        monkeypatch, [WRIST, CameraSourceConfig.from_url("rtp://0.0.0.0:5600", "nose_camera")]
    )
    try:
        on_detections(MagicMock(channel="detections/onnxruntime", payload=_detections()))

        assert driver.cameras["wrist_camera"]._detection_cache.snapshot() is None
    finally:
        asyncio.run(driver.stop_cameras_async())
