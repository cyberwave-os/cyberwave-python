"""Base class for keyed per-sensor twin namespaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

if TYPE_CHECKING:
    from ..base import Twin


class SensorFamilyNamespace:
    """``twin.<family>`` (one sensor) or ``twin.<familys>[<id>]`` (several)."""

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
        self._public_methods = public_methods
        self._handle_for_key = handle_for_key

    def __getitem__(self, key: str) -> Any:
        return self._handle_for_key(self._twin, key)

    def __getattr__(self, name: str) -> Any:
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
        methods = ", ".join(self._public_methods)
        if keys:
            return (
                f"{type(self).__name__}(sensors={keys!r}; "
                f"methods per sensor: {methods})"
            )
        return f"{type(self).__name__}(sensors=[])"

    def keys(self) -> List[str]:
        return list(
            self._twin.resolve_handler_from_capabilities(self._handler_key).sensor_ids
        )

    def values(self) -> List[Any]:
        return [self[key] for key in self.keys()]

    def items(self) -> List[tuple[str, Any]]:
        return [(key, self[key]) for key in self.keys()]

    def describe(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for key in self.keys():
            out[key] = {
                "sensor_id": key,
                "family": self._family_label,
                "methods": list(self._public_methods),
            }
        return out
