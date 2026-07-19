"""Indexable per-family sensor collection exposed as ``twin.<family>``.

A single object serves both single- and multi-sensor twins:

- ``twin.camera.get_frame()`` — unknown attributes proxy to the first sensor ([0]).
- ``twin.camera[0]`` / ``twin.camera['front']`` / ``twin.camera.front`` — pick a sensor.
- ``len(twin.camera)``, iteration, ``.keys()`` / ``.values()`` / ``.items()`` / ``.describe()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Tuple

if TYPE_CHECKING:
    from ..base import Twin
    from ..capability_resolve import HandlerResolution


class SensorFamily:
    """Array-like collection of same-family sensor handles on a twin."""

    def __init__(
        self,
        twin: "Twin",
        *,
        handler_key: str,
        family_label: str,
        public_methods: Tuple[str, ...],
        handle_for_key: Callable[["Twin", str], Any],
    ) -> None:
        self._twin = twin
        self._handler_key = handler_key
        self._family_label = family_label
        self._public_methods = tuple(public_methods)
        self._handle_for_key = handle_for_key
        # Memoize handles per sensor id so stateful handles (e.g. streaming
        # LiDAR/IMU listeners) keep their state across repeated access and
        # attribute proxying: ``twin.lidar.on_pointcloud(cb)`` and
        # ``twin.lidar.get_pointcloud()`` must reach the same instance.
        self._handle_cache: Dict[str, Any] = {}

    def _resolution(self) -> "HandlerResolution":
        return self._twin.resolve_handler_from_capabilities(self._handler_key)

    def keys(self) -> List[str]:
        return list(self._resolution().sensor_ids)

    def __len__(self) -> int:
        return len(self._resolution().sensor_ids)

    def __contains__(self, key: object) -> bool:
        return key in self.keys()

    def __getitem__(self, key: int | str) -> Any:
        if isinstance(key, bool):
            raise TypeError("sensor index must be int or str, not bool")
        if isinstance(key, int):
            ids = self.keys()
            try:
                key = ids[key]
            except IndexError:
                raise IndexError(
                    f"{self._family_label} index {key} out of range "
                    f"({len(ids)} sensor(s): {ids})"
                ) from None
        cached = self._handle_cache.get(key)
        if cached is not None:
            return cached
        handle = self._handle_for_key(self._twin, key)
        self._handle_cache[key] = handle
        return handle

    def __iter__(self) -> Iterator[Any]:
        return (self[key] for key in self.keys())

    def values(self) -> List[Any]:
        return [self[key] for key in self.keys()]

    def items(self) -> List[Tuple[str, Any]]:
        return [(key, self[key]) for key in self.keys()]

    def __getattr__(self, name: str) -> Any:
        # Never proxy private/dunder names: prevents recursion during
        # construction/copy/pickle and keeps internals private.
        if name.startswith("_"):
            raise AttributeError(name)
        keys = self.keys()
        if name in keys:
            return self[name]
        if not keys:
            raise AttributeError(
                f"'{type(self).__name__}' has no {self._family_label} sensors"
            )
        # Default: proxy to the first sensor handle (index 0).
        return getattr(self[0], name)

    def __dir__(self) -> List[str]:
        names = {n for n in object.__dir__(self) if not n.startswith("_")}
        names.update(self.keys())
        names.update(self._public_methods)
        names.update(("describe", "keys", "values", "items"))
        return sorted(names)

    def __repr__(self) -> str:
        keys = self.keys()
        methods = ", ".join(self._public_methods)
        return (
            f"{type(self).__name__}({self._family_label!r}, sensors={keys!r}; "
            f"methods proxy to [0]: {methods})"
        )

    def _sensor_entry(self, key: str) -> Dict[str, Any] | None:
        for entry in self._twin.capabilities.get("sensors", []):
            if not isinstance(entry, dict):
                continue
            entry_id = str(entry.get("id") or entry.get("name") or "")
            if entry_id == key:
                return dict(entry)
        return None

    def describe(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for key in self.keys():
            entry = self._sensor_entry(key) or {}
            out[key] = {
                "sensor_id": key,
                "type": entry.get("type"),
                "family": self._family_label,
                "handle": type(self[key]).__name__,
                "methods": list(self._public_methods),
            }
        return out
