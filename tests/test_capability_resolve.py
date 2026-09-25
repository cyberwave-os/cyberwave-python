"""Central capability → handle resolution."""

from cyberwave.twin.capability_resolve import resolve_handler_from_capabilities

GO2_CAPS = {
    "can_locomote": True,
    "has_joints": True,
    "sensors": [
        {"id": "lidar_4d", "type": "lidar_4d"},
        {"id": "front_camera", "type": "rgb"},
    ],
}


def test_resolve_lidar_and_camera_for_go2() -> None:
    lidar = resolve_handler_from_capabilities(GO2_CAPS, "lidar")
    camera = resolve_handler_from_capabilities(GO2_CAPS, "camera")
    assert lidar.available
    assert lidar.sensor_ids == ("lidar_4d",)
    assert lidar.default_sensor_id == "lidar_4d"
    assert camera.available
    assert camera.sensor_ids == ("front_camera",)
    assert camera.default_sensor_id == "front_camera"


def test_resolve_locomotion_flag() -> None:
    assert resolve_handler_from_capabilities(GO2_CAPS, "locomotion").available
    assert not resolve_handler_from_capabilities(GO2_CAPS, "flight").available


def test_resolve_unknown_handler() -> None:
    assert not resolve_handler_from_capabilities(GO2_CAPS, "hoverboard").available


def test_resolve_gps_imu_compass() -> None:
    caps = {
        "sensors": [
            {"id": "nav_gps", "type": "gps"},
            {"id": "body_imu", "type": "imu"},
            {"id": "mag", "type": "compass"},
        ]
    }
    assert resolve_handler_from_capabilities(caps, "gps").sensor_ids == ("nav_gps",)
    assert resolve_handler_from_capabilities(caps, "imu").sensor_ids == ("body_imu",)
    assert resolve_handler_from_capabilities(caps, "compass").sensor_ids == ("mag",)
