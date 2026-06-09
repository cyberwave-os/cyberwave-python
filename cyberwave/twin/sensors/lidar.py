"""LiDAR sensor handle (read path PR3; metadata available now)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from ..base import Twin


from ..capability_resolve import _is_lidar_type

__all__ = ["LidarSensorHandle", "_is_lidar_type"]


class LidarSensorHandle:
    """Per-sensor LiDAR façade bound to a twin and sensor id."""

    def __init__(self, twin: "Twin", sensor_id: str) -> None:
        self._twin = twin
        self.sensor_id = sensor_id

    def __repr__(self) -> str:
        from ..namespaces.lidar import LIDAR_HANDLE_PUBLIC_METHODS

        methods = ", ".join(LIDAR_HANDLE_PUBLIC_METHODS)
        return f"{type(self).__name__}(sensor_id={self.sensor_id!r}; {methods})"

    def __dir__(self) -> list[str]:
        from ..namespaces.lidar import LIDAR_HANDLE_PUBLIC_METHODS

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

    def get_scan(self) -> Any:
        """Return the latest point cloud / scan (MQTT inbound — PR3)."""
        raise NotImplementedError(
            "lidar.get_scan() requires MQTT inbound readers (PR3). "
            "Use client.on_lidar(twin_uuid) or twin.subscribe() until then."
        )
