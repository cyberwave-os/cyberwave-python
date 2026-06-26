"""GPS, compass, and IMU twin handle stubs."""

from types import SimpleNamespace

import pytest

from cyberwave.twin.classes import LocomoteTwin
from cyberwave.twin.capability_resolve import resolve_handler_from_capabilities


def _twin(*, sensors: list) -> LocomoteTwin:
    return LocomoteTwin(
        SimpleNamespace(twins=SimpleNamespace()),
        SimpleNamespace(
            uuid="rover",
            name="Rover",
            capabilities={"can_locomote": True, "sensors": sensors},
        ),
    )


def test_resolve_gps_compass_imu_handlers() -> None:
    caps = {
        "sensors": [
            {"id": "gps_main", "type": "gps"},
            {"id": "compass_main", "type": "compass"},
            {"id": "imu_main", "type": "imu"},
        ]
    }
    gps = resolve_handler_from_capabilities(caps, "gps")
    compass = resolve_handler_from_capabilities(caps, "compass")
    imu = resolve_handler_from_capabilities(caps, "imu")
    assert gps.available and gps.default_sensor_id == "gps_main"
    assert compass.available and compass.default_sensor_id == "compass_main"
    assert imu.available and imu.default_sensor_id == "imu_main"


def test_has_sensor_for_navigation_families() -> None:
    twin = _twin(
        sensors=[
            {"id": "gps_main", "type": "gps"},
            {"id": "imu_main", "type": "imu"},
            {"id": "compass_main", "type": "compass"},
        ]
    )
    assert twin.gps.metadata()["type"] == "gps"
    assert twin.has_sensor("gps")
    assert twin.has_sensor("imu")
    assert twin.has_sensor("compass")


def test_read_methods_raise_not_implemented() -> None:
    twin = _twin(sensors=[{"id": "gps_main", "type": "gps"}])
    with pytest.raises(NotImplementedError, match="on_gps"):
        twin.gps.get_fix()

    twin = _twin(sensors=[{"id": "compass_main", "type": "compass"}])
    with pytest.raises(NotImplementedError, match="MQTT inbound"):
        twin.compass.get_heading()


def test_cameras_rejects_gps_sensor_key() -> None:
    twin = _twin(
        sensors=[
            {"id": "gps_main", "type": "gps"},
            {"id": "cam_a", "type": "rgb"},
            {"id": "cam_b", "type": "rgb"},
        ]
    )
    with pytest.raises(KeyError, match="use twin.gps"):
        twin.cameras["gps_main"]
