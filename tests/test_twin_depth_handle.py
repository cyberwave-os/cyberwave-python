"""Depth handle: REST-first frame, MQTT raw fallback, point cloud, callbacks."""

import base64
import warnings

import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from cyberwave.exceptions import CyberwaveError, DepthTransportNotMQTTError
from cyberwave.twin.classes import DepthCameraTwin, LocomoteCameraTwin
from cyberwave.twin.sensors.depth import DepthSensorHandle, _decode_depth


def _depth_twin(get_latest_frame=None):
    callbacks: dict[str, object] = {}

    def subscribe(topic, callback, **kwargs):
        callbacks[topic] = callback

    mqtt = MagicMock()
    mqtt.subscribe = subscribe
    mqtt.connected = True
    twins_api = SimpleNamespace(
        api=None,
        get_latest_frame=get_latest_frame or (lambda *a, **k: None),
    )
    client = SimpleNamespace(
        mqtt=mqtt,
        config=SimpleNamespace(topic_prefix="", runtime_mode="live"),
        twins=twins_api,
    )
    twin = LocomoteCameraTwin(
        client,
        SimpleNamespace(
            uuid="rs",
            name="RealSense",
            capabilities={
                "sensors": [{"id": "depth_camera", "type": "depth"}],
            },
        ),
    )
    return twin, callbacks


def _depth_payload(arr: np.ndarray) -> dict:
    """Build an MQTT ``depth_data`` payload, stamping the array's real dtype.

    float arrays carry metres; ``uint16`` carries millimetres.
    """
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape
    return {
        "type": "depth_data",
        "data": {
            "depth_binary": base64.b64encode(arr.tobytes()).decode(),
            "width": w,
            "height": h,
            "dtype": arr.dtype.name,
        },
    }


def test_decode_depth_new_format() -> None:
    arr = (np.arange(6, dtype=np.float32) / 10.0).reshape(2, 3)  # metres
    out = _decode_depth(_depth_payload(arr))
    assert out.shape == (2, 3)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, arr)


def test_decode_depth_uint16_dtype_preserved() -> None:
    arr = np.array([[0, 500], [1000, 2000]], dtype=np.uint16)  # millimetres
    out = _decode_depth(_depth_payload(arr))
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, arr)


def test_decode_depth_legacy_format() -> None:
    # Bare-string legacy shape carried no dtype → decoded as uint16 (millimetres).
    arr = np.arange(6, dtype=np.uint16).reshape(2, 3)
    payload = {
        "type": "depth_data",
        "data": base64.b64encode(arr.tobytes()).decode(),
        "width": 3,
        "height": 2,
    }
    np.testing.assert_array_equal(_decode_depth(payload), arr)


def test_decode_depth_ignores_wrong_type_and_unknown_data() -> None:
    assert _decode_depth({"type": "pointcloud", "data": "x"}) is None
    assert _decode_depth({"type": "depth_data", "data": 123}) is None
    assert _decode_depth({"type": "depth_data"}) is None


def test_depth_get_frame_raw_falls_back_to_mqtt_when_rest_empty() -> None:
    twin, callbacks = _depth_twin(get_latest_frame=lambda *a, **k: None)
    handle = twin.camera[0]
    assert isinstance(handle, DepthSensorHandle)
    handle.on_update(lambda v: None)  # attaches the /depth listener
    arr = (np.arange(6, dtype=np.float32) / 10.0).reshape(2, 3)  # metres
    callbacks["cyberwave/twin/rs/depth"](_depth_payload(arr))
    out = handle.get_frame(raw=True, timeout=0.0)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, arr)
    assert handle._depth_using_rest is False


def test_depth_get_frame_mqtt_float_is_absolute_metres() -> None:
    """MQTT float depth is absolute **metres**; processed frames pass the values
    straight through (non-finite / ≤0 → 0.0)."""
    twin, callbacks = _depth_twin(get_latest_frame=lambda *a, **k: None)
    handle = twin.camera[0]
    handle.on_update(lambda v: None)
    arr = np.array([[0.0, 0.3, 0.611], [1.04, 2.5, np.nan]], dtype=np.float32)
    callbacks["cyberwave/twin/rs/depth"](_depth_payload(arr))
    out = handle.get_frame(timeout=0.0)
    assert out.dtype == np.float32
    assert out[0, 1] == pytest.approx(0.3)
    assert out[0, 2] == pytest.approx(0.611)
    assert out[0, 0] == pytest.approx(0.0)  # 0 m = invalid
    assert out[1, 2] == pytest.approx(0.0)  # NaN = invalid → 0.0


def test_depth_get_frame_mqtt_uint16_is_millimetres_to_metres() -> None:
    """MQTT ``uint16`` depth is millimetres → metres via ``/1000`` (0 = invalid)."""
    twin, callbacks = _depth_twin(get_latest_frame=lambda *a, **k: None)
    handle = twin.camera[0]
    handle.on_update(lambda v: None)
    arr = np.array([[0, 300, 611], [1040, 2500, 3000]], dtype=np.uint16)
    callbacks["cyberwave/twin/rs/depth"](_depth_payload(arr))
    out = handle.get_frame(timeout=0.0)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, arr.astype(np.float32) / 1000.0)
    assert out[0, 0] == pytest.approx(0.0)  # 0 mm = invalid
    assert out[0, 2] == pytest.approx(0.611)  # 611 mm = 0.611 m


def test_depth_get_frame_uses_rest_when_available() -> None:
    # A 1x1 white JPEG so _decode_frame can decode it.
    cv2 = pytest.importorskip("cv2")

    jpeg = cv2.imencode(".jpg", np.zeros((1, 1, 3), dtype=np.uint8))[1].tobytes()
    twin, _ = _depth_twin(get_latest_frame=lambda *a, **k: jpeg)
    handle = twin.camera[0]
    out = handle.get_frame(format="bytes")
    assert isinstance(out, (bytes, bytearray))
    assert handle._depth_using_rest is True


def test_depth_get_frame_rest_grayscale_to_metres() -> None:
    """REST grayscale image is mapped onto the sensor depth range (default 0.1–5.0 m)."""
    cv2 = pytest.importorskip("cv2")

    gray = np.full((8, 8, 3), 128, dtype=np.uint8)
    jpeg = cv2.imencode(".jpg", gray)[1].tobytes()
    twin, _ = _depth_twin(get_latest_frame=lambda *a, **k: jpeg)
    handle = twin.camera[0]
    out = handle.get_frame(format="numpy")  # processed metres (default)
    assert out.dtype == np.float32
    assert out.shape == (8, 8)
    expected = 0.1 + (128.0 / 255.0) * (5.0 - 0.1)
    np.testing.assert_allclose(out, expected, atol=0.1)
    assert handle._depth_using_rest is True


def test_depth_get_frame_rest_raw_returns_grayscale() -> None:
    """raw=True over REST collapses the RGB image to a single-channel uint8 grayscale."""
    cv2 = pytest.importorskip("cv2")

    gray = np.full((8, 8, 3), 200, dtype=np.uint8)
    jpeg = cv2.imencode(".jpg", gray)[1].tobytes()
    twin, _ = _depth_twin(get_latest_frame=lambda *a, **k: jpeg)
    handle = twin.camera[0]
    out = handle.get_frame(format="numpy", raw=True)
    assert out.ndim == 2  # single channel
    assert out.shape == (8, 8)
    assert out.dtype == np.uint8
    assert abs(int(out.mean()) - 200) <= 2  # JPEG rounding
    assert handle._depth_using_rest is True


def test_depth_range_from_capabilities() -> None:
    twin, _ = _depth_twin()
    twin._data.capabilities = {
        "sensors": [
            {
                "id": "depth_camera",
                "type": "depth",
                "parameters": {"min_depth": 0.4, "max_depth": 20.0},
            }
        ]
    }
    handle = twin.camera[0]
    assert handle._depth_range_m() == (0.4, 20.0)


def test_depth_range_defaults_when_missing_or_invalid() -> None:
    twin, _ = _depth_twin()
    handle = twin.camera[0]
    # No parameters -> defaults.
    assert handle._depth_range_m() == (0.1, 5.0)
    # Invalid (max <= min) -> defaults.
    twin._data.capabilities = {
        "sensors": [
            {
                "id": "depth_camera",
                "type": "depth",
                "parameters": {"min_depth": 5, "max_depth": 1},
            }
        ]
    }
    assert handle._depth_range_m() == (0.1, 5.0)


def test_depth_mqtt_rejects_pil_format() -> None:
    twin, callbacks = _depth_twin(get_latest_frame=lambda *a, **k: None)
    handle = twin.camera[0]
    handle.on_update(lambda v: None)
    arr = np.arange(6, dtype=np.uint16).reshape(2, 3)
    callbacks["cyberwave/twin/rs/depth"](_depth_payload(arr))
    with pytest.raises(CyberwaveError, match="format"):
        handle.get_frame(format="pil", source="mqtt", timeout=0.0)


def test_depth_get_pointcloud() -> None:
    twin, callbacks = _depth_twin()
    handle = twin.camera[0]
    handle.on_pointcloud(lambda v: None)
    pts = np.array([[1, 2, 3]], dtype=np.float32)
    callbacks["cyberwave/twin/rs/pointcloud"](
        {
            "type": "pointcloud",
            "data": base64.b64encode(pts.tobytes()).decode(),
        }
    )
    out = handle.get_pointcloud(timeout=0.0)
    assert out.shape == (1, 3)


def test_depth_get_pointcloud_xyzrgb_stride() -> None:
    """Driver sends XYZRGB (stride 6); SDK must return N×3 XYZ, not 2N×3."""
    twin, callbacks = _depth_twin()
    handle = twin.camera[0]
    handle.on_pointcloud(lambda v: None)
    # 2 points, each [x, y, z, r, g, b]
    raw = np.array(
        [[1, 2, 3, 0.9, 0.1, 0.5], [4, 5, 6, 0.2, 0.8, 0.3]], dtype=np.float32
    )
    callbacks["cyberwave/twin/rs/pointcloud"](
        {
            "type": "pointcloud",
            "data": base64.b64encode(raw.tobytes()).decode(),
            "point_count": 2,
            "point_stride": 6,
        }
    )
    out = handle.get_pointcloud(timeout=0.0)
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, raw[:, :3])


def test_on_update_raises_when_transport_is_rest() -> None:
    """on_update must raise DepthTransportNotMQTTError once REST transport is pinned."""
    cv2 = pytest.importorskip("cv2")
    jpeg = cv2.imencode(".jpg", np.zeros((1, 1, 3), dtype=np.uint8))[1].tobytes()
    twin, _ = _depth_twin(get_latest_frame=lambda *a, **k: jpeg)
    handle = twin.camera[0]
    # Pin the transport to REST by calling get_frame successfully.
    handle.get_frame(format="bytes")
    assert handle._depth_using_rest is True
    with pytest.raises(DepthTransportNotMQTTError):
        handle.on_update(lambda v: None)


def test_on_update_allowed_before_transport_is_pinned() -> None:
    """on_update must succeed when transport has not yet been determined."""
    twin, _ = _depth_twin(get_latest_frame=lambda *a, **k: None)
    handle = twin.camera[0]
    assert handle._depth_using_rest is None
    sub = handle.on_update(lambda v: None)
    sub.cancel()


def test_depth_only_twin_camera_is_depth_handle() -> None:
    twin, _ = _depth_twin()
    assert isinstance(twin.camera[0], DepthSensorHandle)


def test_rgb_only_twin_camera_is_plain_camera_handle() -> None:
    from cyberwave.twin.sensors.camera import TwinCameraHandle

    twin, _ = _depth_twin()
    twin._data.capabilities = {"sensors": [{"id": "rgb0", "type": "rgb"}]}
    handle = twin.camera[0]
    assert type(handle) is TwinCameraHandle


def _depth_camera_twin(callbacks_out):
    def subscribe(topic, callback, **kwargs):
        callbacks_out[topic] = callback

    mqtt = MagicMock()
    mqtt.subscribe = subscribe
    mqtt.connected = True
    client = SimpleNamespace(
        mqtt=mqtt,
        config=SimpleNamespace(topic_prefix="", runtime_mode="live"),
        twins=SimpleNamespace(api=None, get_latest_frame=lambda *a, **k: None),
    )
    return DepthCameraTwin(
        client,
        SimpleNamespace(
            uuid="rs",
            name="RealSense",
            capabilities={"sensors": [{"id": "depth_camera", "type": "depth"}]},
        ),
    )


def test_depthcameratwin_get_point_cloud_delegates() -> None:
    callbacks: dict[str, object] = {}
    twin = _depth_camera_twin(callbacks)
    # Attach the listener via the same handle get_point_cloud() delegates to
    # (the default imaging handle), then publish.
    twin._depth_sensor_handle().on_pointcloud(lambda v: None)
    pts = np.array([[1, 2, 3]], dtype=np.float32)
    callbacks["cyberwave/twin/rs/pointcloud"](
        {
            "type": "pointcloud",
            "data": base64.b64encode(pts.tobytes()).decode(),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        out = twin.get_point_cloud(timeout=0.0)
    assert out.shape == (1, 3)
