"""Tests for Twin.capture_frame, Twin.capture_frames, and TwinCameraHandle."""

import os
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.twin_patch import patch_twin

from cyberwave.exceptions import CyberwaveError
from cyberwave.twin import CameraTwin, Twin, TwinCameraHandle, _decode_frame

FAKE_JPEG = b"\xff\xd8fake-jpeg-payload\xff\xd9"


def _make_twin(cls=CameraTwin, source_type=None):
    twins_manager = MagicMock()
    twins_manager.get_latest_frame.return_value = FAKE_JPEG
    if source_type is not None:
        client = SimpleNamespace(
            twins=twins_manager,
            config=SimpleNamespace(source_type=source_type),
        )
    else:
        client = SimpleNamespace(
            twins=twins_manager,
            config=SimpleNamespace(runtime_mode="live", source_type="tele"),
        )
    caps = {"sensors": [{"id": "cam", "type": "rgb", "name": "cam"}]}
    twin = cls(
        client,
        SimpleNamespace(uuid="twin-uuid", name="TestTwin", capabilities=caps),
    )
    def _capture_side_effect(format: str = "numpy", **kwargs: object):
        if format == "path":
            import tempfile

            dest = kwargs.get("path")
            if dest is None:
                fd, dest = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
            else:
                parent = os.path.dirname(os.path.abspath(dest))
                if parent:
                    os.makedirs(parent, exist_ok=True)
            with open(dest, "wb") as f:
                f.write(FAKE_JPEG)
            return os.path.abspath(dest)
        return FAKE_JPEG

    if hasattr(twin, "get_frame"):
        twin.get_frame = MagicMock(side_effect=_capture_side_effect)  # type: ignore[method-assign]
        twin.camera.get_frame = twin.get_frame  # type: ignore[method-assign, union-attr]
    return twin, twins_manager


# ---------------------------------------------------------------------------
# _decode_frame unit tests
# ---------------------------------------------------------------------------


class TestDecodeFrameBytes:
    def test_returns_raw_bytes(self):
        assert _decode_frame(FAKE_JPEG, "bytes") == FAKE_JPEG


class TestDecodeFramePath:
    def test_writes_to_temp_file(self):
        path = _decode_frame(FAKE_JPEG, "path")
        try:
            assert os.path.isfile(path)
            with open(path, "rb") as f:
                assert f.read() == FAKE_JPEG
        finally:
            os.unlink(path)

    def test_file_has_jpg_extension(self):
        path = _decode_frame(FAKE_JPEG, "path")
        try:
            assert path.endswith(".jpg")
        finally:
            os.unlink(path)


class TestDecodeFrameNumpy:
    def test_decodes_with_cv2(self):
        mock_np = MagicMock()
        mock_cv2 = MagicMock()
        mock_arr = MagicMock()
        sentinel_frame = MagicMock(name="decoded_frame")
        mock_np.frombuffer.return_value = mock_arr
        mock_np.uint8 = "uint8"
        mock_cv2.imdecode.return_value = sentinel_frame
        mock_cv2.IMREAD_COLOR = 1

        with patch.dict(
            "sys.modules", {"numpy": mock_np, "cv2": mock_cv2}
        ):
            result = _decode_frame(FAKE_JPEG, "numpy")

        mock_np.frombuffer.assert_called_once_with(FAKE_JPEG, dtype="uint8")
        mock_cv2.imdecode.assert_called_once_with(mock_arr, 1)
        assert result is sentinel_frame

    def test_raises_on_decode_failure(self):
        mock_np = MagicMock()
        mock_cv2 = MagicMock()
        mock_np.frombuffer.return_value = MagicMock()
        mock_np.uint8 = "uint8"
        mock_cv2.imdecode.return_value = None
        mock_cv2.IMREAD_COLOR = 1

        with patch.dict("sys.modules", {"numpy": mock_np, "cv2": mock_cv2}):
            with pytest.raises(CyberwaveError, match="Failed to decode"):
                _decode_frame(FAKE_JPEG, "numpy")

    def test_raises_when_numpy_missing(self):
        with patch.dict("sys.modules", {"numpy": None, "cv2": None}):
            with pytest.raises(CyberwaveError, match="numpy"):
                _decode_frame(FAKE_JPEG, "numpy")


class TestDecodeFramePil:
    def test_decodes_with_pil(self):
        mock_image_mod = MagicMock()
        sentinel_image = MagicMock(name="pil_image")
        mock_image_mod.open.return_value = sentinel_image

        with patch.dict(
            "sys.modules",
            {"PIL": MagicMock(Image=mock_image_mod), "PIL.Image": mock_image_mod},
        ):
            result = _decode_frame(FAKE_JPEG, "pil")

        mock_image_mod.open.assert_called_once()
        assert result is sentinel_image

    def test_raises_when_pillow_missing(self):
        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            with pytest.raises(CyberwaveError, match="Pillow"):
                _decode_frame(FAKE_JPEG, "pil")


class TestDecodeFrameUnknownFormat:
    def test_raises_for_unknown_format(self):
        with pytest.raises(CyberwaveError, match="Unknown format"):
            _decode_frame(FAKE_JPEG, "bmp")


# ---------------------------------------------------------------------------
# Twin.capture_frame
# ---------------------------------------------------------------------------


class TestTwinCaptureFrame:
    def test_bytes_format(self):
        twin, mgr = _make_twin()
        assert twin.capture_frame("bytes") == FAKE_JPEG

    def test_path_format_is_default(self):
        twin, _mgr = _make_twin()
        path = twin.capture_frame()
        try:
            assert os.path.isfile(path)
            with open(path, "rb") as f:
                assert f.read() == FAKE_JPEG
        finally:
            os.unlink(path)

    def test_passes_sensor_id_and_mock(self):
        twin, _mgr = _make_twin()
        twin.capture_frame("bytes", sensor_id="wrist", mock=True)
        twin.get_frame.assert_called_once_with(
            "bytes",
            source="local",
            sensor_id="wrist",
            mock=True,
        )

    def test_wraps_unavailable_capture(self):
        twin, _mgr = _make_twin()
        twin.get_frame = MagicMock(return_value=None)  # type: ignore[method-assign]
        with pytest.raises(CyberwaveError, match="No frame available"):
            twin.capture_frame("bytes")


# ---------------------------------------------------------------------------
# Twin.get_frames
# ---------------------------------------------------------------------------


class TestTwinGetFrames:
    def test_bytes_returns_list(self):
        twin, _mgr = _make_twin()
        result = twin.get_frames(3, interval_ms=0, format="bytes")
        assert result == [FAKE_JPEG] * 3
        assert twin.get_frame.call_count == 3

    def test_path_returns_folder_with_numbered_jpegs(self):
        twin, _mgr = _make_twin()
        folder = twin.get_frames(2, interval_ms=0)
        try:
            assert os.path.isdir(folder)
            files = sorted(os.listdir(folder))
            assert files == ["frame_0000.jpg", "frame_0001.jpg"]
            for fname in files:
                with open(os.path.join(folder, fname), "rb") as f:
                    assert f.read() == FAKE_JPEG
        finally:
            for fname in os.listdir(folder):
                os.unlink(os.path.join(folder, fname))
            os.rmdir(folder)

    def test_rejects_count_less_than_one(self):
        twin, _ = _make_twin()
        with pytest.raises(ValueError, match="count must be >= 1"):
            twin.get_frames(0)

    def test_passes_sensor_id(self):
        twin, _mgr = _make_twin()
        twin.get_frames(1, interval_ms=0, format="bytes", sensor_id="front")
        twin.get_frame.assert_called_with(
            "bytes",
            source="cloud",
            sensor_id="front",
            mock=False,
            idx=0,
            max_age_ms=None,
            zenoh_timeout_s=3.0,
            edge_timeout_s=5.0,
        )

    def test_capture_frames_deprecated_delegates(self):
        twin, _mgr = _make_twin()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = twin.capture_frames(2, interval_ms=0, format="bytes")
        assert result == [FAKE_JPEG] * 2


# ---------------------------------------------------------------------------
# CameraTwin inherits unified capture_frame
# ---------------------------------------------------------------------------


class TestCameraTwinUnified:
    def test_capture_frame_bytes(self):
        twin, _ = _make_twin(CameraTwin)
        assert twin.capture_frame("bytes") == FAKE_JPEG

    def test_capture_frame_path(self):
        twin, _ = _make_twin(CameraTwin)
        path = twin.capture_frame()
        try:
            assert os.path.isfile(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TwinCameraHandle (twin.camera)
# ---------------------------------------------------------------------------


class TestTwinCameraHandle:
    def test_read_defaults_to_numpy(self):
        twin, _ = _make_twin()
        with patch.object(twin.camera, "get_frame", return_value="frame") as mock_gf:
            result = twin.camera.read()
        mock_gf.assert_called_once_with("numpy", source="local")
        assert result == "frame"

    def test_read_passes_format(self):
        twin, _ = _make_twin()
        with patch.object(twin.camera, "get_frame", return_value="img") as mock_gf:
            twin.camera.read("pil", sensor_id="top")
        mock_gf.assert_called_once_with(
            "pil", source="local", sensor_id="top"
        )

    def test_get_frame_path_without_path_writes_temp_jpeg(self):
        twin, mgr = _make_twin()
        real_get_frame = TwinCameraHandle.get_frame
        with patch_twin(
            "sensors.camera._decode_frame",
            return_value="/tmp/cyberwave_test.jpg",
        ):
            mgr.get_latest_frame.return_value = FAKE_JPEG
            result = real_get_frame(twin.camera, "path", source="cloud")
        assert result == os.path.abspath("/tmp/cyberwave_test.jpg")

    def test_get_frame_path_with_dest_writes_file(self, tmp_path):
        twin, mgr = _make_twin()
        dest = str(tmp_path / "out.jpg")
        temp = str(tmp_path / "temp.jpg")
        with open(temp, "wb") as f:
            f.write(FAKE_JPEG)
        real_get_frame = TwinCameraHandle.get_frame
        with patch_twin(
            "sensors.camera._decode_frame",
            return_value=temp,
        ):
            mgr.get_latest_frame.return_value = FAKE_JPEG
            result = real_get_frame(
                twin.camera, "path", path=dest, source="cloud"
            )
        assert result == os.path.abspath(dest)
        with open(dest, "rb") as f:
            assert f.read() == FAKE_JPEG

    def test_snapshot_without_path_uses_get_frame_path(self):
        twin, _ = _make_twin()
        with patch.object(
            twin.camera, "get_frame", return_value="/tmp/snap.jpg"
        ) as mock_gf:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                result = twin.camera.snapshot()
        mock_gf.assert_called_once_with("path", path=None, source="local")
        assert result == "/tmp/snap.jpg"

    def test_snapshot_with_path_writes_file(self, tmp_path):
        twin, _ = _make_twin()
        dest = str(tmp_path / "out.jpg")
        mock_gf = MagicMock(return_value=dest)
        twin.camera.get_frame = mock_gf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = twin.camera.snapshot(dest)
        assert result == dest
        mock_gf.assert_called_once_with("path", path=dest, source="local")

    def test_base_twin_has_no_stream(self):
        client = SimpleNamespace(twins=MagicMock())
        twin = Twin(client, SimpleNamespace(uuid="twin-uuid", name="TestTwin"))
        assert not hasattr(twin, "stream")

    def test_stream_delegates_to_start_streaming(self):
        twin, _ = _make_twin(CameraTwin)
        with patch.object(twin, "start_streaming") as mock_stream:
            twin.camera.stream(fps=15, camera_id=2)
        mock_stream.assert_called_once_with(fps=15, camera_id=2)

    def test_camera_property_returns_same_handle(self):
        twin, _ = _make_twin()
        assert twin.camera is twin.camera
