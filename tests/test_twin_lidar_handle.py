"""LiDAR handle and namespace tests."""

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from cyberwave.twin.classes import LocomoteCameraTwin


def test_single_lidar_exposes_lidar_family() -> None:
    twin = LocomoteCameraTwin(
        SimpleNamespace(twins=SimpleNamespace()),
        SimpleNamespace(
            uuid="go2",
            name="Go2",
            capabilities={
                "can_locomote": True,
                "sensors": [{"id": "lidar_4d", "type": "lidar_4d"}],
            },
        ),
    )
    assert twin.lidar.keys() == ["lidar_4d"]
    assert twin.lidar[0].sensor_id == "lidar_4d"


def test_multi_lidar_exposes_lidar_family() -> None:
    twin = LocomoteCameraTwin(
        SimpleNamespace(twins=SimpleNamespace()),
        SimpleNamespace(
            uuid="go2",
            name="Go2",
            capabilities={
                "can_locomote": True,
                "sensors": [
                    {"id": "lidar_a", "type": "lidar_4d"},
                    {"id": "lidar_b", "type": "lidar_4d"},
                ],
            },
        ),
    )
    assert twin.lidar.keys() == ["lidar_a", "lidar_b"]


def test_single_camera_exposes_camera_family() -> None:
    twin = LocomoteCameraTwin(
        SimpleNamespace(twins=SimpleNamespace()),
        SimpleNamespace(
            uuid="go2",
            capabilities={"sensors": [{"id": "front_camera", "type": "rgb"}]},
        ),
    )
    assert twin.camera[0]._sensor_id == "front_camera"
    assert twin.camera.keys() == ["front_camera"]


def test_multi_camera_uses_camera_family() -> None:
    twin = LocomoteCameraTwin(
        SimpleNamespace(twins=SimpleNamespace()),
        SimpleNamespace(
            uuid="go2",
            capabilities={
                "sensors": [
                    {"id": "cam_a", "type": "rgb"},
                    {"id": "cam_b", "type": "rgb"},
                ],
            },
        ),
    )
    assert twin.camera.keys() == ["cam_a", "cam_b"]
    assert twin.camera[0]._sensor_id == "cam_a"


def test_camera_family_attribute_access_and_discovery() -> None:
    twin = LocomoteCameraTwin(
        SimpleNamespace(twins=SimpleNamespace()),
        SimpleNamespace(
            uuid="rs",
            capabilities={
                "sensors": [
                    {"id": "color_camera", "type": "rgb", "name": "color_camera"},
                    {"id": "depth_camera", "type": "depth", "name": "depth_camera"},
                ],
            },
        ),
    )
    assert twin.camera.color_camera is not twin.camera.depth_camera
    assert hasattr(twin.camera.color_camera, "get_frame")
    assert "color_camera" in dir(twin.camera)
    assert "depth_camera" in dir(twin.camera)
    assert "describe" in dir(twin.camera)
    assert "get_frame" in dir(twin.camera.color_camera)
    info = twin.camera.describe()
    assert set(info) == {"color_camera", "depth_camera"}
    from cyberwave.twin.sensors.camera import CAMERA_HANDLE_PUBLIC_METHODS

    assert info["color_camera"]["methods"] == list(CAMERA_HANDLE_PUBLIC_METHODS)
    assert "get_frame" in repr(twin.camera.color_camera)


def test_multi_lidar_uses_lidar_family() -> None:
    twin = LocomoteCameraTwin(
        SimpleNamespace(twins=SimpleNamespace()),
        SimpleNamespace(
            uuid="go2",
            capabilities={
                "sensors": [
                    {"id": "lidar_a", "type": "lidar_4d"},
                    {"id": "lidar_b", "type": "lidar_4d"},
                ],
            },
        ),
    )
    assert twin.lidar.keys() == ["lidar_a", "lidar_b"]
    assert twin.lidar[1].sensor_id == "lidar_b"


def test_lidar_family_rejects_imaging_sensor_key() -> None:
    twin = LocomoteCameraTwin(
        SimpleNamespace(twins=SimpleNamespace()),
        SimpleNamespace(
            uuid="go2",
            name="Go2",
            capabilities={
                "can_locomote": True,
                "sensors": [
                    {"id": "lidar_a", "type": "lidar_4d"},
                    {"id": "lidar_b", "type": "lidar_4d"},
                    {"id": "front_camera", "type": "rgb"},
                ],
            },
        ),
    )
    with pytest.raises(KeyError, match="not a LiDAR"):
        twin.lidar["front_camera"]


def test_camera_family_rejects_lidar_sensor_key() -> None:
    twin = LocomoteCameraTwin(
        SimpleNamespace(twins=SimpleNamespace()),
        SimpleNamespace(
            uuid="go2",
            name="Go2",
            capabilities={
                "can_locomote": True,
                "sensors": [
                    {"id": "lidar_4d", "type": "lidar_4d"},
                    {"id": "cam_a", "type": "rgb"},
                    {"id": "cam_b", "type": "rgb"},
                ],
            },
        ),
    )
    with pytest.raises(KeyError, match="use twin.lidar"):
        twin.camera["lidar_4d"]


def _lidar_twin():
    callbacks: dict[str, object] = {}

    def subscribe(topic, callback, **kwargs):
        callbacks[topic] = callback

    mqtt = MagicMock()
    mqtt.subscribe = subscribe
    mqtt.connected = True
    client = SimpleNamespace(
        mqtt=mqtt,
        config=SimpleNamespace(topic_prefix=""),
        twins=SimpleNamespace(api=None),
    )
    twin = LocomoteCameraTwin(
        client,
        SimpleNamespace(
            uuid="go2",
            name="Go2",
            capabilities={
                "can_locomote": True,
                "sensors": [{"id": "lidar_4d", "type": "lidar_4d"}],
            },
        ),
    )
    return twin, callbacks


def _pc_payload(points: np.ndarray) -> dict:
    return {
        "type": "pointcloud",
        "data": base64.b64encode(points.astype(np.float32).tobytes()).decode(),
    }


def test_lidar_get_pointcloud_returns_nx3() -> None:
    twin, callbacks = _lidar_twin()
    handle = twin.lidar
    received: list = []
    handle.on_pointcloud(received.append)
    pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    callbacks["cyberwave/twin/go2/pointcloud"](_pc_payload(pts))
    out = handle.get_pointcloud(timeout=0.0)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, pts)
    assert len(received) == 1


def test_lidar_has_no_get_scan() -> None:
    twin, _ = _lidar_twin()
    assert not hasattr(twin.lidar, "get_scan")
