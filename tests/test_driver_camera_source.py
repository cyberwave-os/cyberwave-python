"""Camera source resolution from twin metadata (``cyberwave.driver.sensors.camera_source``)."""

from __future__ import annotations

import pytest

from cyberwave.driver.sensors import camera_source as cs
from cyberwave.driver.sensors.camera_source import (
    CameraConfigurationError,
    CameraHardwareError,
    CameraSourceConfig,
)

RGB_SENSORS = [{"id": "color_camera", "type": "rgb"}]
RGBD_SENSORS = [
    {"id": "color_camera", "type": "rgb"},
    {"id": "depth_camera", "type": "depth"},
]


@pytest.fixture(autouse=True)
def _image_has_every_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "realsense_available", lambda: True)
    monkeypatch.setattr(cs, "gstreamer_available", lambda: True)


@pytest.fixture(autouse=True)
def _clean_metadata_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "VIDEO_DEVICE",
        "SERIAL_NUMBER",
        "IS_DEPTH_CAMERA",
        "RESOLUTION",
        "FPS",
        "DEPTH_FPS",
        "GST_SOURCE",
        "H264_PASSTHROUGH",
    ):
        monkeypatch.delenv(f"CYBERWAVE_METADATA_{name}", raising=False)


def _by_id(tmp_path, *entries: str) -> str:
    by_id = tmp_path / "by-id"
    by_id.mkdir()
    for entry in entries:
        (by_id / entry).touch()
    return str(by_id)


class TestParsers:
    def test_index_strings_become_ints(self) -> None:
        assert cs.parse_camera_id("2") == 2

    @pytest.mark.parametrize("value", ["/dev/video2", "rtsp://cam/stream"])
    def test_paths_and_urls_pass_through(self, value: str) -> None:
        assert cs.parse_camera_id(value) == value

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("256x192", (256, 192)), ("640X480", (640, 480)), (" 1920 x 1080 ", (1920, 1080))],
    )
    def test_resolution(self, value: str, expected: tuple[int, int]) -> None:
        assert cs.parse_resolution(value) == expected

    @pytest.mark.parametrize("value", [None, "", "640", "axb", "0x480", "-1x2"])
    def test_bad_resolution_is_ignored(self, value: str | None) -> None:
        assert cs.parse_resolution(value) is None

    @pytest.mark.parametrize("sensor_type", ["depth", "depth_camera"])
    def test_both_depth_spellings_count(self, sensor_type: str) -> None:
        assert cs.sensors_declare_depth([{"type": sensor_type}])

    @pytest.mark.parametrize(
        "camera_id", ["/dev/v4l/by-id/usb-x-video-index0", "rtsp://cam", "046322252081"]
    )
    def test_stable_identifiers(self, camera_id: str) -> None:
        assert cs.is_stable_device_identifier(camera_id)

    @pytest.mark.parametrize("camera_id", [0, "/dev/video0"])
    def test_positional_identifiers(self, camera_id: int | str) -> None:
        assert not cs.is_stable_device_identifier(camera_id)


class TestResolveUvcSerial:
    def test_prefers_capture_node(self, tmp_path) -> None:
        by_id = _by_id(
            tmp_path,
            "usb-046d_C920_ABC123-video-index1",
            "usb-046d_C920_ABC123-video-index0",
        )
        assert cs.resolve_uvc_serial("ABC123", by_id).endswith("ABC123-video-index0")

    def test_serial_prefix_of_another_device_does_not_match(self, tmp_path) -> None:
        by_id = _by_id(tmp_path, "usb-046d_C920_ABC1234-video-index0")
        assert cs.resolve_uvc_serial("ABC123", by_id) is None

    def test_missing_directory(self) -> None:
        assert cs.resolve_uvc_serial("ABC123", "/nonexistent/by-id") is None


class TestFromSensors:
    def test_unconfigured_rgb_twin(self) -> None:
        config = CameraSourceConfig.from_sensors(RGB_SENSORS)

        assert config.backend == "cv2"
        assert config.camera_id == 0
        assert not config.pinned
        assert config.camera_name == "color_camera"
        assert config.fps == 30

    def test_rgbd_twin_uses_realsense_and_keeps_serial_zeros(self, monkeypatch) -> None:
        monkeypatch.setenv("CYBERWAVE_METADATA_SERIAL_NUMBER", "046322252081")

        config = CameraSourceConfig.from_sensors(RGBD_SENSORS)

        assert config.backend == "realsense"
        assert config.serial_number == "046322252081"
        assert config.pinned
        assert config.depth_sensor_name == "depth_camera"

    def test_is_depth_camera_overrides_sensors(self, monkeypatch) -> None:
        monkeypatch.setenv("CYBERWAVE_METADATA_IS_DEPTH_CAMERA", "false")

        assert CameraSourceConfig.from_sensors(RGBD_SENSORS).backend == "cv2"

    def test_rtsp_url_is_pinned(self, monkeypatch) -> None:
        monkeypatch.setenv("CYBERWAVE_METADATA_VIDEO_DEVICE", "rtsp://cam/stream")

        config = CameraSourceConfig.from_sensors(RGB_SENSORS)

        assert config.camera_id == "rtsp://cam/stream"
        assert config.pinned

    def test_dev_video_allows_fallback(self, monkeypatch) -> None:
        monkeypatch.setenv("CYBERWAVE_METADATA_VIDEO_DEVICE", "/dev/video2")

        assert not CameraSourceConfig.from_sensors(RGB_SENSORS).pinned

    def test_uvc_serial_resolves_to_by_id(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cs, "V4L_BY_ID_DIR", _by_id(tmp_path, "usb-x_ABC123-video-index0"))
        monkeypatch.setenv("CYBERWAVE_METADATA_SERIAL_NUMBER", "ABC123")

        config = CameraSourceConfig.from_sensors(RGB_SENSORS)

        assert str(config.camera_id).endswith("usb-x_ABC123-video-index0")
        assert config.pinned

    def test_unknown_uvc_serial_raises(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cs, "V4L_BY_ID_DIR", _by_id(tmp_path))
        monkeypatch.setenv("CYBERWAVE_METADATA_SERIAL_NUMBER", "NOPE")

        with pytest.raises(CameraHardwareError, match="NOPE"):
            CameraSourceConfig.from_sensors(RGB_SENSORS)

    def test_serial_without_by_id_dir_uses_video_device(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cs, "V4L_BY_ID_DIR", str(tmp_path / "missing"))
        monkeypatch.setenv("CYBERWAVE_METADATA_SERIAL_NUMBER", "ABC123")
        monkeypatch.setenv("CYBERWAVE_METADATA_VIDEO_DEVICE", "1")

        config = CameraSourceConfig.from_sensors(RGB_SENSORS)

        assert config.camera_id == 1
        assert not config.pinned

    def test_stream_url_outranks_serial(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cs, "V4L_BY_ID_DIR", _by_id(tmp_path, "usb-x_ABC123-video-index0"))
        monkeypatch.setenv("CYBERWAVE_METADATA_SERIAL_NUMBER", "ABC123")
        monkeypatch.setenv("CYBERWAVE_METADATA_VIDEO_DEVICE", "http://host:8080/mjpeg")

        assert CameraSourceConfig.from_sensors(RGB_SENSORS).camera_id == "http://host:8080/mjpeg"

    def test_gst_source_selects_pre_encoded_backend(self, monkeypatch) -> None:
        monkeypatch.setenv(
            "CYBERWAVE_METADATA_GST_SOURCE", "v4l2src device=/dev/video2 ! video/x-h264"
        )

        config = CameraSourceConfig.from_sensors(RGB_SENSORS)

        assert config.is_pre_encoded
        assert config.gst_source == "v4l2src device=/dev/video2 ! video/x-h264"
        assert config.pinned

    def test_rtsp_passthrough(self, monkeypatch) -> None:
        monkeypatch.setenv("CYBERWAVE_METADATA_VIDEO_DEVICE", "rtsp://cam/stream")
        monkeypatch.setenv("CYBERWAVE_METADATA_H264_PASSTHROUGH", "true")

        config = CameraSourceConfig.from_sensors(RGB_SENSORS)

        assert config.is_pre_encoded
        assert 'rtspsrc location="rtsp://cam/stream"' in (config.gst_source or "")

    @pytest.mark.parametrize(
        ("url", "fragment"),
        [
            ("rtp://0.0.0.0:5600", "rtph264depay"),
            ("udp://239.0.0.1:5000", "tsdemux"),
            ("tcp://10.0.0.5:8000", "tcpclientsrc host=10.0.0.5 port=8000"),
        ],
    )
    def test_passthrough_for_network_streams(self, monkeypatch, url: str, fragment: str) -> None:
        monkeypatch.setenv("CYBERWAVE_METADATA_VIDEO_DEVICE", url)
        monkeypatch.setenv("CYBERWAVE_METADATA_H264_PASSTHROUGH", "true")

        config = CameraSourceConfig.from_sensors(RGB_SENSORS)

        assert config.is_pre_encoded and config.pinned
        assert fragment in (config.gst_source or "")

    def test_passthrough_for_unsupported_scheme_is_a_config_error(self, monkeypatch) -> None:
        monkeypatch.setenv("CYBERWAVE_METADATA_VIDEO_DEVICE", "http://cam/mjpeg")
        monkeypatch.setenv("CYBERWAVE_METADATA_H264_PASSTHROUGH", "true")

        with pytest.raises(CameraConfigurationError, match="http"):
            CameraSourceConfig.from_sensors(RGB_SENSORS)

    def test_http_without_passthrough_still_goes_through_opencv(self, monkeypatch) -> None:
        monkeypatch.setenv("CYBERWAVE_METADATA_VIDEO_DEVICE", "http://cam/mjpeg")

        assert CameraSourceConfig.from_sensors(RGB_SENSORS).backend == "cv2"

    def test_resolution_and_rates(self, monkeypatch) -> None:
        monkeypatch.setenv("CYBERWAVE_METADATA_RESOLUTION", "256x192")
        monkeypatch.setenv("CYBERWAVE_METADATA_FPS", "15")
        monkeypatch.setenv("CYBERWAVE_METADATA_DEPTH_FPS", "nope")

        config = CameraSourceConfig.from_sensors(RGB_SENSORS)

        assert config.resolution == (256, 192)
        assert config.fps == 15
        assert config.depth_fps == 30

    def test_from_env_reads_twin_json(self, tmp_path, monkeypatch) -> None:
        twin_json = tmp_path / "twin.json"
        twin_json.write_text('{"capabilities": {"sensors": [{"id": "wrist", "type": "rgb"}]}}')
        monkeypatch.setenv("CYBERWAVE_TWIN_JSON_FILE", str(twin_json))

        assert CameraSourceConfig.from_env().camera_name == "wrist"


class TestImageWithoutOptionalBackends:
    """The standard image has neither librealsense nor GStreamer."""

    def test_depth_twin_streams_color_over_v4l2(self, monkeypatch) -> None:
        monkeypatch.setattr(cs, "realsense_available", lambda: False)
        monkeypatch.setenv("CYBERWAVE_METADATA_SERIAL_NUMBER", "046322252081")

        config = CameraSourceConfig.from_sensors(RGBD_SENSORS)

        assert config.backend == "cv2"
        assert config.serial_number is None
        assert not config.pinned

    def test_explicit_depth_request_fails_clearly(self, monkeypatch) -> None:
        monkeypatch.setattr(cs, "realsense_available", lambda: False)
        monkeypatch.setenv("CYBERWAVE_METADATA_IS_DEPTH_CAMERA", "true")

        with pytest.raises(CameraConfigurationError, match="-realsense"):
            CameraSourceConfig.from_sensors(RGB_SENSORS)

    def test_broken_pyrealsense2_counts_as_missing(self, monkeypatch) -> None:
        import builtins

        monkeypatch.undo()
        real_import = builtins.__import__

        def _broken(name, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "pyrealsense2":
                raise OSError("librealsense2.so: cannot open shared object file")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _broken)

        assert cs.realsense_available() is False


class TestDeclaredSources:
    def test_push_is_pre_encoded_appsrc(self) -> None:
        config = CameraSourceConfig.push(camera_name="front_camera", fps=15)

        assert (config.backend, config.gst_source, config.camera_name, config.fps) == (
            "gstreamer", "appsrc", "front_camera", 15
        )
        assert config.is_pre_encoded and config.pinned

    def test_from_url_maps_the_url(self) -> None:
        config = CameraSourceConfig.from_url("rtp://0.0.0.0:5600", camera_name="cam")

        assert config.is_pre_encoded and config.pinned
        assert "rtph264depay" in (config.gst_source or "")

    def test_from_url_rejects_unsupported_schemes(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            CameraSourceConfig.from_url("http://cam/mjpeg")

    def test_describe_hides_credentials(self) -> None:
        text = CameraSourceConfig.from_url("rtsp://admin:s3cr3t@10.0.0.9/ch1").describe()

        assert "s3cr3t" not in text and "***@10.0.0.9" in text


class TestDeclaredSources:
    """The factories a driver uses in CAMERA_SOURCE / CAMERAS."""

    def test_usb_index_is_positional(self) -> None:
        config = cs.CameraSourceConfig.usb(0, "wrist_camera")
        assert (config.backend, config.camera_id, config.pinned) == ("cv2", 0, False)

    def test_usb_by_id_link_and_serial_pin_the_camera(self) -> None:
        by_id = cs.CameraSourceConfig.usb("/dev/v4l/by-id/usb-Logitech-video-index0")
        by_serial = cs.CameraSourceConfig.usb(serial_number="0123ABCD")
        assert by_id.pinned and by_serial.pinned
        assert by_serial.serial_number == "0123ABCD", "kept as a string, leading zero intact"

    def test_usb_serial_resolves_when_the_camera_opens(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cs, "V4L_BY_ID_DIR", str(tmp_path))
        config = cs.CameraSourceConfig.usb(serial_number="0123ABCD", camera_name="cam")

        with pytest.raises(cs.CameraHardwareError, match="0123ABCD"):
            cs.resolve_usb_serial(config)

        (tmp_path / "usb-Acme_Cam_0123ABCD-video-index0").touch()
        resolved = cs.resolve_usb_serial(config)
        assert resolved.camera_id == str(tmp_path / "usb-Acme_Cam_0123ABCD-video-index0")
        assert resolved.pinned

    def test_other_sources_are_not_resolved(self) -> None:
        config = cs.CameraSourceConfig.from_url("rtsp://10.0.0.9/stream")
        assert cs.resolve_usb_serial(config) is config

    def test_url_passthrough_needs_no_decode(self) -> None:
        rtp = cs.CameraSourceConfig.from_url("rtp://0.0.0.0:5600", "nose_camera")
        assert rtp.is_pre_encoded and rtp.pinned
        assert rtp.gst_source.startswith("udpsrc address=0.0.0.0 port=5600")

    def test_url_decoded_by_opencv(self) -> None:
        config = cs.CameraSourceConfig.from_url(
            "http://10.0.0.9/mjpg/video.mjpg", "gate_camera", decode=True, resolution=(1280, 720)
        )
        assert (config.backend, config.camera_id) == ("cv2", "http://10.0.0.9/mjpg/video.mjpg")
        assert config.pinned and not config.is_pre_encoded
        assert config.resolution == (1280, 720)


def test_describe_never_shows_url_credentials() -> None:
    decoded = cs.CameraSourceConfig.from_url("rtsp://admin:s3cret@cam.lan/s", decode=True)
    forwarded = cs.CameraSourceConfig.from_url("rtsp://admin:s3cret@cam.lan/s")

    assert "s3cret" not in decoded.describe() and "s3cret" not in forwarded.describe()
    assert "rtsp://***@cam.lan/s" in decoded.describe()
