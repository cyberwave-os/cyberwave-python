"""Per-sensor twin handles keyed by sensor id."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..capability_resolve import (
    _is_compass_type,
    _is_flashlight_type,
    _is_gps_type,
    _is_imu_type,
    _is_lidar_type,
)
from .camera import CAMERA_HANDLE_PUBLIC_METHODS, TwinCameraHandle  # noqa: F401
from .compass import COMPASS_HANDLE_PUBLIC_METHODS, CompassSensorHandle
from .depth import DepthSensorHandle
from .family import SensorFamily
from .flashlight import FLASHLIGHT_HANDLE_PUBLIC_METHODS, FlashlightSensorHandle
from .gps import GPS_HANDLE_PUBLIC_METHODS, GpsSensorHandle
from .imu import IMU_HANDLE_PUBLIC_METHODS, ImuSensorHandle
from .lidar import LIDAR_HANDLE_PUBLIC_METHODS, LidarSensorHandle
from .base import BaseSensorHandle
from .rgb import RGBSensorHandle

if TYPE_CHECKING:
    from ..base import Twin
    from .camera import TwinCameraHandle

_SENSOR_TYPES = {
    "rgb": RGBSensorHandle,
    "camera": RGBSensorHandle,
    "depth": DepthSensorHandle,
}

READ_SENSOR_METHODS: dict[str, tuple[str, ...]] = {
    "lidar": LIDAR_HANDLE_PUBLIC_METHODS,
    "gps": GPS_HANDLE_PUBLIC_METHODS,
    "compass": COMPASS_HANDLE_PUBLIC_METHODS,
    "imu": IMU_HANDLE_PUBLIC_METHODS,
    "flashlight": FLASHLIGHT_HANDLE_PUBLIC_METHODS,
}

_NON_IMAGING_REDIRECTS: tuple[tuple[str, str, str], ...] = (
    ("lidar", "lidar", "LiDAR"),
    ("gps", "gps", "GPS"),
    ("compass", "compass", "compass"),
    ("imu", "imu", "IMU"),
    ("flashlight", "flashlight", "flashlight"),
)


def _redirect_hint(twin: "Twin", *, singular: str, key: str, label: str) -> str:
    return f"Sensor '{key}' is {label}; use twin.{singular}['{key}']"


def sensor_handle_for_key(twin: "Twin", key: str) -> "TwinCameraHandle":
    for entry in twin.capabilities.get("sensors", []):
        entry_id = str(entry.get("id") or entry.get("name") or "")
        if entry_id != key:
            continue
        sensor_type = str(entry.get("type") or "rgb").lower()
        for handler, singular, label in _NON_IMAGING_REDIRECTS:
            pred = {
                "lidar": _is_lidar_type,
                "gps": _is_gps_type,
                "compass": _is_compass_type,
                "imu": _is_imu_type,
                "flashlight": _is_flashlight_type,
            }[handler]
            if pred(sensor_type):
                raise KeyError(
                    _redirect_hint(twin, singular=singular, key=key, label=label)
                )
        cls = _SENSOR_TYPES.get(sensor_type, RGBSensorHandle)
        return cls(twin, entry_id)  # type: ignore[return-value]
    raise KeyError(f"No sensor '{key}' on twin {twin.uuid}")


def _handle_for_key(
    twin: "Twin",
    key: str,
    *,
    predicate,
    handle_cls: type,
    not_found_msg: str,
    wrong_type_msg: str,
) -> Any:
    for entry in twin.capabilities.get("sensors", []):
        if not isinstance(entry, dict):
            continue
        entry_id = str(entry.get("id") or entry.get("name") or "")
        if entry_id != key:
            continue
        sensor_type = str(entry.get("type") or "")
        if not predicate(sensor_type):
            raise KeyError(wrong_type_msg.format(key=key, uuid=twin.uuid))
        return handle_cls(twin, entry_id)
    raise KeyError(not_found_msg.format(key=key, uuid=twin.uuid))


def lidar_handle_for_key(twin: "Twin", key: str) -> LidarSensorHandle:
    return _handle_for_key(
        twin,
        key,
        predicate=_is_lidar_type,
        handle_cls=LidarSensorHandle,
        not_found_msg="No LiDAR sensor '{key}' on twin {uuid}",
        wrong_type_msg="Sensor '{key}' is not a LiDAR on twin {uuid}",
    )


def gps_handle_for_key(twin: "Twin", key: str) -> GpsSensorHandle:
    return _handle_for_key(
        twin,
        key,
        predicate=_is_gps_type,
        handle_cls=GpsSensorHandle,
        not_found_msg="No GPS sensor '{key}' on twin {uuid}",
        wrong_type_msg="Sensor '{key}' is not a GPS on twin {uuid}",
    )


def compass_handle_for_key(twin: "Twin", key: str) -> CompassSensorHandle:
    return _handle_for_key(
        twin,
        key,
        predicate=_is_compass_type,
        handle_cls=CompassSensorHandle,
        not_found_msg="No compass sensor '{key}' on twin {uuid}",
        wrong_type_msg="Sensor '{key}' is not a compass on twin {uuid}",
    )


def imu_handle_for_key(twin: "Twin", key: str) -> ImuSensorHandle:
    return _handle_for_key(
        twin,
        key,
        predicate=_is_imu_type,
        handle_cls=ImuSensorHandle,
        not_found_msg="No IMU sensor '{key}' on twin {uuid}",
        wrong_type_msg="Sensor '{key}' is not an IMU on twin {uuid}",
    )


def flashlight_handle_for_key(twin: "Twin", key: str) -> FlashlightSensorHandle:
    return _handle_for_key(
        twin,
        key,
        predicate=_is_flashlight_type,
        handle_cls=FlashlightSensorHandle,
        not_found_msg="No flashlight sensor '{key}' on twin {uuid}",
        wrong_type_msg="Sensor '{key}' is not a flashlight on twin {uuid}",
    )


__all__ = [
    "BaseSensorHandle",
    "RGBSensorHandle",
    "DepthSensorHandle",
    "LidarSensorHandle",
    "GpsSensorHandle",
    "CompassSensorHandle",
    "ImuSensorHandle",
    "FlashlightSensorHandle",
    "sensor_handle_for_key",
    "lidar_handle_for_key",
    "gps_handle_for_key",
    "compass_handle_for_key",
    "imu_handle_for_key",
    "flashlight_handle_for_key",
    "SensorFamily",
    "READ_SENSOR_METHODS",
    "CAMERA_HANDLE_PUBLIC_METHODS",
    "LIDAR_HANDLE_PUBLIC_METHODS",
    "GPS_HANDLE_PUBLIC_METHODS",
    "COMPASS_HANDLE_PUBLIC_METHODS",
    "IMU_HANDLE_PUBLIC_METHODS",
    "FLASHLIGHT_HANDLE_PUBLIC_METHODS",
]
