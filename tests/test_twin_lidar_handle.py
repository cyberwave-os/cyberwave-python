"""LiDAR handle and namespace tests."""

from types import SimpleNamespace

import pytest

from cyberwave.twin.classes import LocomoteCameraTwin


def test_single_lidar_has_no_lidars_namespace() -> None:
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
    with pytest.raises(AttributeError, match="no attribute 'lidars'"):
        twin.lidars


def test_multi_lidar_exposes_lidars_namespace() -> None:
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
    assert twin.lidars.keys() == ["lidar_a", "lidar_b"]


def test_single_camera_has_no_cameras_namespace() -> None:
    twin = LocomoteCameraTwin(
        SimpleNamespace(twins=SimpleNamespace()),
        SimpleNamespace(
            uuid="go2",
            capabilities={"sensors": [{"id": "front_camera", "type": "rgb"}]},
        ),
    )
    assert twin.camera._sensor_id == "front_camera"
    with pytest.raises(AttributeError, match="no attribute 'cameras'"):
        twin.cameras


def test_multi_camera_has_no_camera_singular() -> None:
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
    assert twin.cameras.keys() == ["cam_a", "cam_b"]
    with pytest.raises(AttributeError, match="no attribute 'camera'"):
        twin.camera


def test_cameras_namespace_attribute_access_and_discovery() -> None:
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
    assert twin.cameras.color_camera is not twin.cameras.depth_camera
    assert hasattr(twin.cameras.color_camera, "get_frame")
    assert "color_camera" in dir(twin.cameras)
    assert "depth_camera" in dir(twin.cameras)
    assert "describe" in dir(twin.cameras)
    assert "get_frame" in dir(twin.cameras.color_camera)
    info = twin.cameras.describe()
    assert set(info) == {"color_camera", "depth_camera"}
    assert info["color_camera"]["methods"] == [
        "get_frame",
        "get_frames",
        "stream",
        "read",
        "snapshot",
    ]
    assert "get_frame" in repr(twin.cameras.color_camera)


def test_multi_lidar_has_no_lidar_singular() -> None:
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
    with pytest.raises(AttributeError, match="no attribute 'lidar'"):
        twin.lidar


def test_lidar_namespace_rejects_imaging_sensor_key() -> None:
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
        twin.lidars["front_camera"]


def test_cameras_rejects_lidar_sensor_key() -> None:
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
        twin.cameras["lidar_4d"]
