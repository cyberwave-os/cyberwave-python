"""Base sensor handle — binds a sensor id to the twin camera façade."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import Twin
    from .camera import TwinCameraHandle


def _camera_handle_class() -> type:
    from .camera import TwinCameraHandle

    return TwinCameraHandle


class BaseSensorHandle:
    """Sensor-scoped camera handle (subclasses :class:`TwinCameraHandle`)."""

    def __new__(cls, twin: "Twin", sensor_id: str) -> "TwinCameraHandle":
        handle_cls = _camera_handle_class()
        return handle_cls(twin, sensor_id=sensor_id)
