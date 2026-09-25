"""Compass sensor handle (read path — MQTT inbound not yet wired)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from ..simulation_support import SimLevel, simulation_level

if TYPE_CHECKING:
    from ..base import Twin

__all__ = ["CompassSensorHandle", "COMPASS_HANDLE_PUBLIC_METHODS", "_is_compass_type"]

COMPASS_HANDLE_PUBLIC_METHODS: tuple[str, ...] = ("metadata", "get_heading")


def _is_compass_type(sensor_type: str) -> bool:
    return sensor_type.lower() == "compass"


class CompassSensorHandle:
    """Per-sensor compass façade bound to a twin and sensor id."""

    def __init__(self, twin: "Twin", sensor_id: str) -> None:
        self._twin = twin
        self.sensor_id = sensor_id

    def __repr__(self) -> str:
        methods = ", ".join(COMPASS_HANDLE_PUBLIC_METHODS)
        return f"{type(self).__name__}(sensor_id={self.sensor_id!r}; {methods})"

    def __dir__(self) -> list[str]:
        names = {n for n in object.__dir__(self) if not n.startswith("_")}
        names.update(COMPASS_HANDLE_PUBLIC_METHODS)
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

    @simulation_level(SimLevel.UNSUPPORTED)
    def get_heading(self) -> Any:
        """Return the latest compass heading.

        WIP: not yet implemented — raises ``NotImplementedError`` in live mode
        too (the MQTT inbound reader isn't wired up yet), not just in simulation.
        """
        raise NotImplementedError(
            "twin.compass.get_heading() is not yet implemented."
        )
