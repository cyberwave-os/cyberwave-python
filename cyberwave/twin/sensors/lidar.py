"""LiDAR sensor handle — point cloud over MQTT ``/pointcloud``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from ..capability_resolve import _is_lidar_type
from .pointcloud import PointCloudCapableMixin

if TYPE_CHECKING:
    from ..base import Twin

__all__ = ["LidarSensorHandle", "LIDAR_HANDLE_PUBLIC_METHODS", "_is_lidar_type"]

LIDAR_HANDLE_PUBLIC_METHODS: tuple[str, ...] = (
    "metadata",
    "get_pointcloud",
    "on_pointcloud",
)


class LidarSensorHandle(PointCloudCapableMixin):
    """Per-sensor LiDAR façade bound to a twin and sensor id."""

    def __init__(self, twin: "Twin", sensor_id: str) -> None:
        super().__init__(twin, sensor_id)

    def __repr__(self) -> str:
        methods = ", ".join(LIDAR_HANDLE_PUBLIC_METHODS)
        return f"{type(self).__name__}(sensor_id={self.sensor_id!r}; {methods})"

    def __dir__(self) -> list[str]:
        names = {n for n in object.__dir__(self) if not n.startswith("_")}
        names.update(LIDAR_HANDLE_PUBLIC_METHODS)
        return sorted(names)

    def metadata(self) -> Dict[str, Any]:
        """Capability entry for this sensor (model, FOV, range, rates, …)."""
        for entry in self._twin.capabilities.get("sensors", []):
            if not isinstance(entry, dict):
                continue
            entry_id = str(entry.get("id") or entry.get("name") or "")
            if entry_id == self.sensor_id:
                return dict(entry)
        return {}
