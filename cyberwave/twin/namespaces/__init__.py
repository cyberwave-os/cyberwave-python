"""Keyed sensor namespaces exposed on :class:`~cyberwave.twin.base.Twin`."""

from .base import SensorFamilyNamespace
from .camera import CamerasNamespace
from .compass import COMPASS_HANDLE_PUBLIC_METHODS, CompassesNamespace
from .gps import GPS_HANDLE_PUBLIC_METHODS, GpssNamespace
from .flashlight import FLASHLIGHT_HANDLE_PUBLIC_METHODS, FlashlightsNamespace
from .imu import IMU_HANDLE_PUBLIC_METHODS, ImusNamespace
from .lidar import LIDAR_HANDLE_PUBLIC_METHODS, LidarsNamespace

READ_SENSOR_METHODS: dict[str, tuple[str, ...]] = {
    "lidar": LIDAR_HANDLE_PUBLIC_METHODS,
    "gps": GPS_HANDLE_PUBLIC_METHODS,
    "compass": COMPASS_HANDLE_PUBLIC_METHODS,
    "imu": IMU_HANDLE_PUBLIC_METHODS,
    "flashlight": FLASHLIGHT_HANDLE_PUBLIC_METHODS,
}

__all__ = [
    "SensorFamilyNamespace",
    "CamerasNamespace",
    "LidarsNamespace",
    "GpssNamespace",
    "CompassesNamespace",
    "ImusNamespace",
    "FlashlightsNamespace",
    "LIDAR_HANDLE_PUBLIC_METHODS",
    "FLASHLIGHT_HANDLE_PUBLIC_METHODS",
    "GPS_HANDLE_PUBLIC_METHODS",
    "COMPASS_HANDLE_PUBLIC_METHODS",
    "IMU_HANDLE_PUBLIC_METHODS",
    "READ_SENSOR_METHODS",
]
