"""Flashlight sensor handle — illumination via twin MQTT commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ..base import Twin

__all__ = ["FlashlightSensorHandle"]


class FlashlightSensorHandle:
    """Per-sensor flashlight façade bound to a twin and sensor id."""

    def __init__(self, twin: "Twin", sensor_id: str) -> None:
        self._twin = twin
        self.sensor_id = sensor_id

    def __repr__(self) -> str:
        from ..namespaces.flashlight import FLASHLIGHT_HANDLE_PUBLIC_METHODS

        methods = ", ".join(FLASHLIGHT_HANDLE_PUBLIC_METHODS)
        return f"{type(self).__name__}(sensor_id={self.sensor_id!r}; {methods})"

    def __dir__(self) -> list[str]:
        from ..namespaces.flashlight import FLASHLIGHT_HANDLE_PUBLIC_METHODS

        names = {n for n in object.__dir__(self) if not n.startswith("_")}
        names.update(FLASHLIGHT_HANDLE_PUBLIC_METHODS)
        return sorted(names)

    def metadata(self) -> Dict[str, Any]:
        """Capability entry for this sensor."""
        for entry in self._twin.capabilities.get("sensors", []):
            if not isinstance(entry, dict):
                continue
            entry_id = str(entry.get("id") or entry.get("name") or "")
            if entry_id == self.sensor_id:
                return dict(entry)
        return {}

    def set(self, *, on: bool = True, source_type: Optional[str] = None) -> None:
        """Turn illumination on or off (``lights_on`` / ``lights_off`` MQTT commands)."""
        command = "lights_on" if on else "lights_off"
        self._twin._prepare_outbound_command()
        resolved = self._twin._resolve_topic_and_payload(
            command=command,
            data={"on": on, "sensor_id": self.sensor_id},
            source_type=source_type,
        )
        self._twin._publish_resolved(resolved)
