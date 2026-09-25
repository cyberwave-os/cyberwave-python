"""MQTT command catalog handle for twins (invocation only)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..manifest.driver_config import (
    TWIN_COMMAND_TOPIC_SLUG,
    supported_mqtt_commands,
)
from .command_factory import bind_catalog_commands

if TYPE_CHECKING:
    from .base import Twin


class TwinCommandsHandle:
    """Invoke MQTT catalog commands bound on this twin.

    Use :meth:`get_schema` / :meth:`get_supported_commands` for command-catalog
    introspection without reaching for :attr:`~cyberwave.twin.base.Twin.driver`.
    Full MQTT + Zenoh catalogs and :meth:`~cyberwave.twin.driver.TwinDriverHandle.set_schema`
    remain on ``twin.driver``.

    At construction, every name in ``commands.supported`` is bound as
    ``twin.commands.<name>(...)``.
    """

    def __init__(self, twin: Twin) -> None:
        self._twin = twin
        self._bound_catalog_commands: list[str] = []
        self._command_routing: dict[str, dict[str, Any]] = {}
        bind_catalog_commands(self)

    def get_schema(self, *, force_refresh: bool = False) -> dict[str, Any]:
        """MQTT command catalog (topics, ``commands.supported``, specs).

        Shortcut for :meth:`~cyberwave.twin.driver.TwinDriverHandle.get_mqtt_schema`.
        """
        return self._twin.driver.get_mqtt_schema(force_refresh=force_refresh)

    def get_supported_commands(self) -> list[str]:
        """Command names from ``commands.supported`` in the MQTT catalog."""
        return self._twin.driver.get_supported_commands()

    def __dir__(self) -> list[str]:
        names = list(object.__dir__(self))
        names.extend(self._bound_catalog_commands)
        return sorted(set(names))

    def describe_section(self) -> dict[str, Any]:
        """Agent-facing summary of MQTT catalog command invocation."""
        catalog_methods = list(self._bound_catalog_commands)
        supported = self.get_supported_commands()
        introspection = ["get_schema", "get_supported_commands"]

        section: dict[str, Any] = {
            "access": "twin.commands",
            "role": "mqtt_catalog_command_invocation",
            "methods": introspection + catalog_methods,
            "catalog_methods": catalog_methods,
            "catalog_commands": catalog_methods,
            "supported_commands": supported,
            "catalog_introspection": "twin.commands.get_schema(), twin.commands.get_supported_commands(); twin.driver for set_schema and Zenoh",
            "publish": {
                "topic_slug": TWIN_COMMAND_TOPIC_SLUG,
                "envelope_fields": ["source_type", "command", "data", "timestamp"],
                "call_pattern": "twin.commands.<command>(data={}, **kwargs)",
            },
        }
        command_routing = getattr(self, "_command_routing", None)
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
