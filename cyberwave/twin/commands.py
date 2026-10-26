"""MQTT command catalog handle for twins (invocation only)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..manifest.driver_config import TWIN_COMMAND_TOPIC_SLUG
from .command_factory import bind_catalog_commands

if TYPE_CHECKING:
    from ..commands import (
        CommandResult,
        TwinCommandGateway,
        TwinControlGrantManager,
        TwinControlPolicyManager,
    )
    from .base import Twin


class TwinCommandsHandle:
    """Invoke MQTT catalog commands bound on this twin.

    Use :meth:`get_schema` / :meth:`get_supported_commands` for command-catalog
    introspection without reaching for :attr:`~cyberwave.twin.base.Twin.driver`.
    Full MQTT + Zenoh catalogs and :meth:`~cyberwave.twin.driver.TwinDriverHandle.set_schema`
    remain on ``twin.driver``.

    At construction, every name in ``commands.supported`` is bound as
    ``twin.commands.<name>(...)`` (a direct MQTT publish).

    Governed commands go through the platform command gateway instead:
    :meth:`send` (policy-checked, audited, optionally awaiting the driver's
    reply), :meth:`audit`, :attr:`control_policy` and :attr:`grants`.
    :meth:`request` sends a command over MQTT v5 request/response directly
    (ungoverned twins only). ``send``, ``request``, ``audit``,
    ``control_policy``, ``grants`` and ``gateway`` are therefore never bound
    as catalog commands; publish a catalog command with one of those names
    through ``twin.publish_command(name, data)``.
    """

    def __init__(self, twin: Twin) -> None:
        self._twin = twin
        self._bound_catalog_commands: list[str] = []
        self._command_routing: dict[str, dict[str, Any]] = {}
        self._gateway_handle: TwinCommandGateway | None = None
        bind_catalog_commands(self)

    # -- command gateway (REST) -------------------------------------------------

    @property
    def gateway(self) -> TwinCommandGateway:
        """The platform command gateway bound to this twin."""
        if self._gateway_handle is None:
            from ..commands import TwinCommandGateway

            self._gateway_handle = TwinCommandGateway(self._twin.client, self._twin.uuid)
        return self._gateway_handle

    def send(
        self,
        command: str,
        data: dict[str, Any] | None = None,
        *,
        timeout_s: float = 0,
        mode: str | None = None,
    ) -> CommandResult:
        """Issue *command* through the governed command gateway.

        See :meth:`cyberwave.commands.TwinCommandGateway.send`.
        """
        return self.gateway.send(command, data, timeout_s=timeout_s, mode=mode)

    def audit(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        decision: str | None = None,
        command: str | None = None,
    ) -> dict[str, Any]:
        """This twin's command audit page, newest first."""
        return self.gateway.audit(
            limit=limit, offset=offset, decision=decision, command=command
        )

    @property
    def control_policy(self) -> TwinControlPolicyManager:
        """``get()`` / ``put(...)`` / ``delete()`` this twin's control policy."""
        return self.gateway.control_policy

    @property
    def grants(self) -> TwinControlGrantManager:
        """``list()`` / ``create(...)`` / ``revoke(uuid)`` operator grants."""
        return self.gateway.grants

    # -- MQTT v5 request/response ---------------------------------------------

    def request(
        self,
        command: str,
        data: dict[str, Any] | None = None,
        *,
        timeout_s: float = 5.0,
        source_type: str | None = None,
    ) -> dict[str, Any]:
        """Send *command* over MQTT v5 and return the driver's reply payload.

        The command is validated against the MQTT catalog and resolved like
        ``twin.commands.<name>()`` (see
        :meth:`~cyberwave.twin.transport.TwinTransportMixin._request_command`);
        reply correlation is
        :meth:`cyberwave.mqtt.CyberwaveMQTTClient.request_command`'s. Governed
        twins refuse direct publishes: use :meth:`send` for those.
        """
        return self._twin._request_command(
            command,
            data,
            timeout_s=timeout_s,
            source_type=source_type,
        )

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
        gateway_methods = ["send", "request", "audit", "control_policy", "grants"]

        section: dict[str, Any] = {
            "access": "twin.commands",
            "role": "mqtt_catalog_command_invocation",
            "methods": introspection + gateway_methods + catalog_methods,
            "gateway_methods": gateway_methods,
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
