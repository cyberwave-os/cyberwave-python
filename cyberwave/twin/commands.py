"""MQTT command catalog handle for twins."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from ..manifest.driver_config import (
    JOINT_UPDATE_TOPIC_SLUG,
    TWIN_COMMAND_TOPIC_SLUG,
    command_specs,
    extract_mqtt_bundle_from_metadata,
    has_joint_update_topic,
    supported_mqtt_commands,
)
from ..exceptions import CyberwaveError
from ..manifest.cw_driver import (
    merge_mqtt_into_metadata,
    resolve_mqtt_bundle_from_driver_config,
)
from ._helpers import _get_twin_metadata
from .command_factory import bind_catalog_commands, rebind_catalog_commands

if TYPE_CHECKING:
    from .base import Twin


class TwinCommandsHandle:
    """Access the compiled MQTT interface catalog stored on the twin.

    The catalog lives in ``twin.metadata`` (copied from the asset at twin
    creation). At construction, every name in ``commands.supported`` from
    :meth:`get_schema` is bound as a method that publishes the standard twin
    command envelope (``source_type``, ``command``, ``data``, ``timestamp``).
    """

    def __init__(self, twin: Twin) -> None:
        self._twin = twin
        self._bound_catalog_commands: list[str] = []
        self._command_routing: dict[str, dict[str, Any]] = {}
        bind_catalog_commands(self)

    def get_schema(self, *, force_refresh: bool = False) -> dict[str, Any]:
        """Return the MQTT command catalog (topics, supported commands, joint_control)."""
        if not force_refresh and getattr(self._twin, "_mqtt_catalog_cache", None) is not None:
            return copy.deepcopy(self._twin._mqtt_catalog_cache)

        bundle = self._load_mqtt_bundle()
        if bundle is None:
            bundle = {"commands": {"supported": []}, "topics": {}}

        schema = copy.deepcopy(bundle)
        self._merge_joint_control(schema)
        self._twin._mqtt_catalog_cache = schema
        return copy.deepcopy(schema)

    def set_schema(
        self,
        driver_config: Union[str, Path, dict[str, Any]],
        *,
        merge: bool = True,
    ) -> dict[str, Any]:
        """Update this twin's MQTT driver catalog from ``cw-driver.yml`` or a compiled bundle.

        Compiles the manifest (when given a file path or cw-driver root dict), persists
        ``metadata["mqtt"]`` on this twin via the platform API, clears the local catalog
        cache, and re-binds catalog-derived ``twin.commands.<name>`` methods.

        Args:
            driver_config: Path to ``cw-driver.yml``, a cw-driver root dict, or a compiled
                ``metadata["mqtt"]`` bundle dict.
            merge: When ``True``, deep-merge into existing twin ``metadata["mqtt"]``; when
                ``False``, replace the whole ``mqtt`` block.

        Returns:
            The compiled MQTT catalog after persistence.

        Raises:
            CyberwaveError: If the REST client cannot update the twin.
            FileNotFoundError: If *driver_config* is a path that does not exist.
            ValueError: If the manifest is invalid.
        """
        bundle = resolve_mqtt_bundle_from_driver_config(driver_config)
        metadata = merge_mqtt_into_metadata(
            _get_twin_metadata(self._twin._data),
            bundle,
            merge=merge,
        )

        twins_api = getattr(self._twin.client, "twins", None)
        if twins_api is None or not hasattr(twins_api, "update"):
            raise CyberwaveError(
                "Cannot persist driver catalog: Cyberwave client has no twins.update API"
            )

        try:
            updated = twins_api.update(self._twin.uuid, metadata=metadata)
        except Exception as exc:
            raise CyberwaveError(
                f"Failed to update twin {self._twin.uuid!r} driver catalog: {exc}"
            ) from exc

        self._twin._data = updated
        self._twin._mqtt_catalog_cache = None
        rebind_catalog_commands(self)
        return self.get_schema(force_refresh=True)

    def _load_mqtt_bundle(self) -> dict[str, Any] | None:
        metadata = _get_twin_metadata(self._twin._data)
        return extract_mqtt_bundle_from_metadata(metadata)

    def _merge_joint_control(self, schema: dict[str, Any]) -> None:
        if not has_joint_update_topic(schema):
            return
        from .capabilities.joints import controllable_joint_names

        names = controllable_joint_names(self._twin)
        if not names:
            return
        schema["joint_control"] = {
            "controllable_joint_names": names,
            "joints": [
                {"name": name}
                for name in names
            ],
        }

    def __dir__(self) -> list[str]:
        names = list(object.__dir__(self))
        names.extend(self._bound_catalog_commands)
        return sorted(set(names))

    def describe_section(self) -> dict[str, Any]:
        """Agent-facing summary of ``twin.commands`` and the MQTT catalog."""
        schema = self.get_schema()
        catalog_methods = list(self._bound_catalog_commands)
        supported = supported_mqtt_commands(schema)
        specs = command_specs(schema)

        topics = schema.get("topics")
        topic_slugs = (
            sorted(str(k) for k in topics.keys())
            if isinstance(topics, dict)
            else []
        )

        mqtt: dict[str, Any] = {
            "supported": supported,
            "specs": specs,
            "topics": topic_slugs,
            "command_topic_slug": TWIN_COMMAND_TOPIC_SLUG,
            "has_joint_update_topic": JOINT_UPDATE_TOPIC_SLUG in set(topic_slugs),
        }
        joint_control = schema.get("joint_control")
        if isinstance(joint_control, dict) and joint_control:
            mqtt["joint_control"] = joint_control

        provenance = schema.get("provenance")
        if isinstance(provenance, dict) and provenance:
            mqtt["provenance"] = provenance

        command_routing = getattr(self, "_command_routing", None)
        section: dict[str, Any] = {
            "access": "twin.commands",
            "methods": ["get_schema", "set_schema", *catalog_methods],
            "catalog_methods": catalog_methods,
            "catalog_commands": catalog_methods,
            "publish": {
                "topic_slug": TWIN_COMMAND_TOPIC_SLUG,
                "envelope_fields": ["source_type", "command", "data", "timestamp"],
                "call_pattern": "twin.commands.<command>(data={}, **kwargs)",
            },
            "mqtt": mqtt,
        }
        if isinstance(command_routing, dict) and command_routing:
            section["command_routing"] = command_routing
        return section

    def __repr__(self) -> str:
        cmds = ", ".join(self._bound_catalog_commands[:8])
        suffix = "…" if len(self._bound_catalog_commands) > 8 else ""
        return (
            f"TwinCommandsHandle(twin={self._twin.uuid!r}, "
            f"catalog_commands=[{cmds}{suffix}])"
        )
