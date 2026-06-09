"""IMU sensor handle (read path — MQTT inbound not yet wired)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from ..base import Twin

__all__ = ["ImuSensorHandle", "_is_imu_type"]


def _is_imu_type(sensor_type: str) -> bool:
    return sensor_type.lower() == "imu"


class ImuSensorHandle:
    """Per-sensor IMU façade bound to a twin and sensor id."""

    def __init__(self, twin: "Twin", sensor_id: str) -> None:
        self._twin = twin
        self.sensor_id = sensor_id

    def __repr__(self) -> str:
        from ..namespaces.imu import IMU_HANDLE_PUBLIC_METHODS

        methods = ", ".join(IMU_HANDLE_PUBLIC_METHODS)
        return f"{type(self).__name__}(sensor_id={self.sensor_id!r}; {methods})"

    def __dir__(self) -> list[str]:
        from ..namespaces.imu import IMU_HANDLE_PUBLIC_METHODS

        names = {n for n in object.__dir__(self) if not n.startswith("_")}
        names.update(IMU_HANDLE_PUBLIC_METHODS)
        return sorted(names)

    def metadata(self) -> Dict[str, Any]:
        """Capability entry for this sensor (rate, axes, frame, …)."""
        for entry in self._twin.capabilities.get("sensors", []):
            if not isinstance(entry, dict):
                continue
            entry_id = str(entry.get("id") or entry.get("name") or "")
            if entry_id == self.sensor_id:
                return dict(entry)
        return {}

    def get_sample(self) -> Any:
        """Return the latest IMU sample (MQTT inbound — not yet wired)."""
        raise NotImplementedError(
            "imu.get_sample() requires MQTT inbound readers. "
            "Use client.on_imu(twin_uuid) or twin.subscribe() until then."
        )
