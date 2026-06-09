"""Imaging namespace — ``twin.cameras[<id>]`` (multi-sensor twins)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from ..sensors.camera import CAMERA_HANDLE_PUBLIC_METHODS, TwinCameraHandle

if TYPE_CHECKING:
    from ..base import Twin


class CamerasNamespace:
    """Keyed access to per-sensor camera handles."""

    def __init__(self, twin: "Twin") -> None:
        self._twin = twin

    def __getitem__(self, key: str) -> TwinCameraHandle:
        from ..sensors import sensor_handle_for_key

        return sensor_handle_for_key(self._twin, key)

    def __getattr__(self, name: str) -> TwinCameraHandle:
        if name in self.keys():
            return self[name]
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute {name!r}; "
            f"known sensors: {', '.join(self.keys()) or '(none)'}"
        )

    def __dir__(self) -> List[str]:
        names = {n for n in object.__dir__(self) if not n.startswith("_")}
        names.update(self.keys())
        names.update(("describe", "items", "values"))
        return sorted(names)

    def __repr__(self) -> str:
        keys = self.keys()
        if keys:
            return (
                f"CamerasNamespace(sensors={keys!r}; "
                f"access via .{keys[0]} or ['{keys[0]}'] when present; "
                f"methods per sensor: {', '.join(CAMERA_HANDLE_PUBLIC_METHODS)})"
            )
        return "CamerasNamespace(sensors=[])"

    def keys(self) -> List[str]:
        return list(self._twin.resolve_handler_from_capabilities("camera").sensor_ids)

    def values(self) -> List[TwinCameraHandle]:
        return [self[key] for key in self.keys()]

    def items(self) -> List[tuple[str, TwinCameraHandle]]:
        return [(key, self[key]) for key in self.keys()]

    def describe(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for key in self.keys():
            entry = self._sensor_entry(key) or {}
            out[key] = {
                "sensor_id": key,
                "type": entry.get("type"),
                "handle": type(self[key]).__name__,
                "methods": list(CAMERA_HANDLE_PUBLIC_METHODS),
            }
        return out

    def _sensor_entry(self, key: str) -> Dict[str, Any] | None:
        for entry in self._twin.capabilities.get("sensors", []):
            if not isinstance(entry, dict):
                continue
            entry_id = str(entry.get("id") or entry.get("name") or "")
            if entry_id == key:
                return dict(entry)
        return None
