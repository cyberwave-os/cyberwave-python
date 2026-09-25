"""Resolve twin handles from the capabilities dict (single entry point)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

_IMAGING_TYPES = frozenset({"rgb", "depth", "camera"})


def _is_lidar_type(sensor_type: str) -> bool:
    return "lidar" in sensor_type.lower()


def _is_gps_type(sensor_type: str) -> bool:
    return sensor_type.lower() == "gps"


def _is_compass_type(sensor_type: str) -> bool:
    return sensor_type.lower() == "compass"


def _is_imu_type(sensor_type: str) -> bool:
    return sensor_type.lower() == "imu"


def _is_flashlight_type(sensor_type: str) -> bool:
    return sensor_type.lower() == "flashlight"

# Boolean capability flags → handle name
_FLAG_BY_HANDLER: dict[str, str] = {
    "locomotion": "can_locomote",
    "flight": "can_fly",
    "gripper": "can_grip",
    "joints": "has_joints",
    "actuate": "can_actuate",
}


@dataclass(frozen=True)
class HandlerResolution:
    """Result of :func:`resolve_handler_from_capabilities`."""

    available: bool
    sensor_ids: tuple[str, ...] = ()
    default_sensor_id: Optional[str] = None
    sensor_entries: tuple[dict[str, Any], ...] = ()

    @property
    def multi_sensor(self) -> bool:
        """True when keyed namespace (``cameras`` / ``lidars``) is appropriate."""
        return len(self.sensor_ids) > 1


def _sensor_entries(capabilities: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = capabilities.get("sensors") or []
    return [s for s in raw if isinstance(s, dict)]


def _entry_id(entry: Mapping[str, Any]) -> Optional[str]:
    for key in ("id", "name", "role"):
        value = entry.get(key)
        if value:
            return str(value)
    return None


def resolve_imaging_sensor_id(
    capabilities: Mapping[str, Any],
    sensor_id: Optional[str],
) -> Optional[str]:
    """Map imaging sensor aliases (name/role) to canonical id for routing."""
    resolution = resolve_handler_from_capabilities(capabilities, "camera")
    if not resolution.available:
        return sensor_id
    if sensor_id is None:
        return resolution.default_sensor_id
    for entry in resolution.sensor_entries:
        aliases = {
            str(value)
            for key in ("id", "name", "role")
            if (value := entry.get(key))
        }
        if sensor_id in aliases:
            return _entry_id(entry)
    return sensor_id


def _resolve_sensor_family(
    capabilities: Mapping[str, Any],
    predicate: Callable[[Mapping[str, Any]], bool],
) -> HandlerResolution:
    entries = [e for e in _sensor_entries(capabilities) if predicate(e)]
    ids = tuple(i for e in entries if (i := _entry_id(e)))
    return HandlerResolution(
        available=bool(ids),
        sensor_ids=ids,
        default_sensor_id=ids[0] if ids else None,
        sensor_entries=tuple(entries),
    )


def resolve_handler_from_capabilities(
    capabilities: Mapping[str, Any],
    handler: str,
) -> HandlerResolution:
    """Whether a grouped handle is available and optional per-sensor metadata.

    Handles:
    - ``camera`` / ``imaging`` — RGB, depth, or camera sensors
    - ``lidar`` — any sensor type string containing ``lidar``
    - ``gps``, ``compass``, ``imu``, ``flashlight`` — exact sensor type match
    - ``sensor`` — any entry in ``capabilities.sensors``
    - ``locomotion``, ``flight``, ``gripper``, ``joints``, ``actuate`` — boolean flags
    """
    caps = capabilities or {}
    key = handler.strip().lower()

    if key in ("camera", "imaging"):
        return _resolve_sensor_family(
            caps,
            predicate=lambda e: str(e.get("type") or "") in _IMAGING_TYPES,
        )
    if key == "lidar":
        return _resolve_sensor_family(
            caps,
            predicate=lambda e: _is_lidar_type(str(e.get("type") or "")),
        )
    if key == "gps":
        return _resolve_sensor_family(
            caps,
            predicate=lambda e: _is_gps_type(str(e.get("type") or "")),
        )
    if key == "compass":
        return _resolve_sensor_family(
            caps,
            predicate=lambda e: _is_compass_type(str(e.get("type") or "")),
        )
    if key == "imu":
        return _resolve_sensor_family(
            caps,
            predicate=lambda e: _is_imu_type(str(e.get("type") or "")),
        )
    if key == "flashlight":
        return _resolve_sensor_family(
            caps,
            predicate=lambda e: _is_flashlight_type(str(e.get("type") or "")),
        )
    if key == "sensor":
        entries = _sensor_entries(caps)
        ids = tuple(i for e in entries if (i := _entry_id(e)))
        return HandlerResolution(
            available=bool(ids),
            sensor_ids=ids,
            default_sensor_id=ids[0] if ids else None,
            sensor_entries=tuple(entries),
        )

    flag_key = _FLAG_BY_HANDLER.get(key)
    if flag_key is not None:
        return HandlerResolution(available=bool(caps.get(flag_key)))

    return HandlerResolution(available=False)
