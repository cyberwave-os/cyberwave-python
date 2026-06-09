"""Per-sensor twin handles keyed by sensor id."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..capability_resolve import (
    _is_compass_type,
    _is_flashlight_type,
    _is_gps_type,
    _is_imu_type,
    _is_lidar_type,
    resolve_handler_from_capabilities,
)
from .camera import TwinCameraHandle  # noqa: F401 — load before imaging subclasses
from .compass import CompassSensorHandle
from .depth import DepthSensorHandle
from .flashlight import FlashlightSensorHandle
from .gps import GpsSensorHandle
from .imu import ImuSensorHandle
from .lidar import LidarSensorHandle
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

_NON_IMAGING_REDIRECTS: tuple[tuple[str, str, str, str], ...] = (
    ("lidar", "lidar", "lidars", "LiDAR"),
    ("gps", "gps", "gpss", "GPS"),
    ("compass", "compass", "compasses", "compass"),
    ("imu", "imu", "imus", "IMU"),
    ("flashlight", "flashlight", "flashlights", "flashlight"),
)


def _redirect_hint(
    twin: "Twin",
    *,
    handler: str,
    singular: str,
    plural: str,
    key: str,
    label: str,
) -> str:
    resolution = resolve_handler_from_capabilities(twin.capabilities, handler)
    hint = (
        f"twin.{plural}['{key}']"
        if resolution.multi_sensor
        else f"twin.{singular}  # sensor {resolution.default_sensor_id!r}"
    )
    return f"Sensor '{key}' is {label}; use {hint}"


def sensor_handle_for_key(twin: "Twin", key: str) -> "TwinCameraHandle":
    for entry in twin.capabilities.get("sensors", []):
        entry_id = str(entry.get("id") or entry.get("name") or "")
        if entry_id != key:
            continue
        sensor_type = str(entry.get("type") or "rgb").lower()
        for handler, singular, plural, label in _NON_IMAGING_REDIRECTS:
            pred = {
                "lidar": _is_lidar_type,
                "gps": _is_gps_type,
                "compass": _is_compass_type,
                "imu": _is_imu_type,
                "flashlight": _is_flashlight_type,
            }[handler]
            if pred(sensor_type):
                raise KeyError(
                    _redirect_hint(
                        twin,
                        handler=handler,
                        singular=singular,
                        plural=plural,
                        key=key,
                        label=label,
                    )
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
]
