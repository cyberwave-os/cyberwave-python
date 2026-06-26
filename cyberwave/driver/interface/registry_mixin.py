"""Interface registry wiring for :class:`~cyberwave.driver.BaseDriver` (Python SDK).

Builds manifests, subscribes MQTT command topics, and (when declared via
``TopicSpec(enable_zenoh=True)`` on publishers) publishes on the SDK
:class:`~cyberwave.data.api.DataBus` using the bound twin UUID.

Also mixed into edge-runtime ``BaseROS2Driver`` drivers; the same registry API
applies, but lifecycle hooks differ (sync vs async).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from typing_extensions import Self

from cyberwave.telemetry.base import BaseTelemetry

from .args import (
    CallbackGroup,
    DriverOperationMode,
    PublisherArgs,
    TopicSpec,
    mqtt_spec,
    zenoh_spec,
)
from .registry import (
    DriverInterfaceRegistry,
    default_management_commands,
    resolve_topic_path,
)
from .source_type_policy import filtered_listener

logger = logging.getLogger(__name__)

ListenerHandler = Callable[[dict[str, Any]], None | Awaitable[None]]


class InterfaceRegistryMixin:
    """Registry build, manifest export, and runtime MQTT + optional Zenoh wiring.

    Used by :class:`~cyberwave.driver.BaseDriver`. Declare topics in
    :meth:`define_interface`; MQTT paths use the cloud broker, Zenoh paths use
    :meth:`_ensure_registry_zenoh_bus` (``cw.data_bus_for(twin_uuid)``).
    """

    REGISTRY_ID: str = ""
    driver_family: str = "python"
    auto_register_interface: bool = True
    TELEMETRY_PUBLISH_RATE_HZ: float = 2.0
    # When True, the built-in management commands (controller-changed, teleoperate,
    # remoteoperate, stop) stay functional but are omitted from the exported
    # catalog. Drivers that don't expose them as robot capabilities opt in.
    HIDE_DEFAULT_MANAGEMENT_COMMANDS_FROM_CATALOG: bool = False

    _interface: DriverInterfaceRegistry
    _telemetry: BaseTelemetry
    _operation_mode: DriverOperationMode
    _wired_mqtt_handlers: list[tuple[str, ListenerHandler]]
    _publisher_tick_counter: int
    _registry_zenoh_bus: Any
    _registry_zenoh_subscriptions: list[Any]

    def _init_interface_registry(self, *, auto_register_interface: bool | None = None) -> None:
        self._interface = DriverInterfaceRegistry()
        self._operation_mode = DriverOperationMode.NO_OP
        self._wired_mqtt_handlers = []
        self._publisher_tick_counter = 0
        self._registry_zenoh_bus = None
        self._registry_zenoh_subscriptions = []
        self._controller_initialized: bool = False
        self._last_controller_uuid: str | None = None
        self._auto_register_interface = (
            auto_register_interface
            if auto_register_interface is not None
            else type(self).auto_register_interface
        )
        self._telemetry = BaseTelemetry(
            publish_payload=self._publish_driver_telemetry_payload,
            snapshot_provider=self._build_driver_telemetry_snapshot,
            source_type="edge",
        )
        self.define_interface_defaults(self._interface)
        self.define_interface(self._interface)

    def define_interface_defaults(self, iface: DriverInterfaceRegistry) -> None:
        """Register built-in management commands and telemetry publisher."""
        default_management_commands(
            iface,
            on_controller_changed=CallbackGroup(self._on_controller_changed_cmd),
            on_teleoperate=CallbackGroup(self._on_teleoperate_cmd),
            on_remoteoperate=CallbackGroup(self._on_remoteoperate_cmd),
            on_stop=CallbackGroup(self._on_stop_cmd),
            catalog_hidden=type(self).HIDE_DEFAULT_MANAGEMENT_COMMANDS_FROM_CATALOG,
        )
        all_modes = frozenset(DriverOperationMode)
        iface.add_publisher(
            TopicSpec(
                namespace="twin",
                leaf="telemetry",
                payload_schema_ref="TwinTelemetryPayload",
                description="Driver lifecycle and health snapshots",
            ),
            CallbackGroup(callback=self._driver_telemetry_publisher_callback),
            publisher=PublisherArgs(rate_hz=type(self).TELEMETRY_PUBLISH_RATE_HZ),
            operation_modes=all_modes,
        )

    def define_interface(self, iface: DriverInterfaceRegistry) -> None:
        """Override to declare MQTT/Zenoh topic specs, commands, and callbacks."""

    @classmethod
    def _class_registry_id(cls) -> str:
        """Registry ID from ``REGISTRY_ID`` (class attribute)."""
        key = getattr(cls, "REGISTRY_ID", None)
        if isinstance(key, str) and key.strip():
            return key.strip()
        return ""

    def _resolved_registry_id(self) -> str:
        """Registry ID for manifest export and telemetry (subclass ``REGISTRY_ID``)."""
        return self.registry_id

    @classmethod
    def _manifest_probe(cls) -> Self:
        """Minimal driver instance for manifest export (no twin, MQTT, or lifecycle)."""
        params: Any = None
        params_cls = getattr(cls, "Params", None)
        if params_cls is not None:
            params = (
                params_cls.from_env()
                if hasattr(params_cls, "from_env")
                else params_cls()
            )
        kwargs: dict[str, Any] = {
            "client": None,
            "auto_register_interface": False,
        }
        init_params = list(inspect.signature(cls.__init__).parameters)[1:]
        if init_params and init_params[0] == "twin":
            return cls(twin=None, params=params, **kwargs)
        return cls(params, twin=None, **kwargs)

    @classmethod
    def get_manifest(
        cls,
        *,
        compiled: bool = False,
        path: str | Path | None = None,
        header_comment: str | None = None,
    ) -> dict[str, Any]:
        """Build cw-driver root dict from :meth:`define_interface` without running the driver.

        Args:
            compiled: Must be ``False``. Compilation happens on the backend via
                :meth:`~cyberwave.twin.driver.TwinDriverHandle.set_schema`.
            path: When set, also write ``cw-driver.yml`` (YAML-shaped root dict).
            header_comment: Optional comment lines at the top of the written file.

        Returns:
            Uncompiled ``cw-driver.yml`` root dict.
        """
        if compiled:
            raise ValueError(
                "compiled driver catalogs are produced by the backend; "
                "use compiled=False and twin.driver.set_schema()"
            )
        probe = cls._manifest_probe()
        if path is not None:
            probe.write_cw_driver_yml(path, header_comment=header_comment)
        return probe.get_driver_manifest(compiled=False)

    def get_driver_manifest(self, *, compiled: bool = False) -> dict[str, Any]:
        """Export uncompiled cw-driver.yml root dict (no callbacks)."""
        if compiled:
            raise ValueError(
                "compiled driver catalogs are produced by the backend; "
                "use compiled=False and twin.driver.set_schema()"
            )
        registry_id = self._resolved_registry_id()
        return self._interface.to_cw_driver_dict(
            registry_id=registry_id,
            driver_family=type(self).driver_family,
        )

    @property
    def manifest(self) -> dict[str, Any]:
        """cw-driver.yml root dict for :meth:`~cyberwave.twin.driver.TwinDriverHandle.set_schema`."""
        return self.get_driver_manifest(compiled=False)

    @property
    def cw_driver(self) -> dict[str, Any]:
        """``cw-driver.yml`` root dict produced from :meth:`define_interface` (not compiled)."""
        return self.get_driver_manifest(compiled=False)

    def write_cw_driver_yml(
        self,
        path: str | Path | None = None,
        *,
        header_comment: str | None = None,
    ) -> Path:
        """Export :attr:`cw_driver` to a ``cw-driver.yml`` file on disk."""
        from .cw_driver import CW_DRIVER_FILE_NAME, dump_cw_driver_yml

        target = Path(path or CW_DRIVER_FILE_NAME)
        comment = header_comment or (
            f"Generated from {type(self).__module__}.{type(self).__qualname__}"
            ".define_interface — prefer editing Python and re-exporting."
        )
        dump_cw_driver_yml(self.cw_driver, target, header_comment=comment)
        logger.info("Wrote %s", target.resolve())
        return target.resolve()

    def register_interface_on_twin(self, *, merge: bool = True) -> dict[str, Any]:
        """Persist this driver's MQTT catalog on the connected twin."""
        twin = self._require_twin()
        schemas = twin.driver.set_schema(self.cw_driver, merge=merge)
        mqtt_schema = schemas.get("mqtt") if isinstance(schemas.get("mqtt"), dict) else {}
        self._emit_driver_info_after_schema(mqtt_schema)
        return schemas

    @property
    def operation_mode(self) -> DriverOperationMode:
        return self._operation_mode

    async def _set_operation_mode(self, mode: DriverOperationMode) -> None:
        if mode == self._operation_mode:
            return
        await self.on_exit_operation()
        self._operation_mode = mode
        if self._is_interface_wired():
            await self._unwire_interface_from_registry()
            await self._wire_interface_from_registry()
        if hasattr(self, "_ros_forward_handles"):
            from ..ros2.ros_publishers import unwire_ros_publishers, wire_ros_publishers

            unwire_ros_publishers(self)
            wire_ros_publishers(self)
        if mode == DriverOperationMode.TELEOP_LOCAL:
            await self.on_enter_teleop_local()
        elif mode == DriverOperationMode.TELEOP_REMOTE:
            await self.on_enter_teleop_remote()
        else:
            await self.on_enter_no_op()
        self._emit_driver_info(operation_mode=mode.value)

    async def on_enter_no_op(self) -> None:
        pass

    async def on_enter_teleop_local(self) -> None:
        pass

    async def on_enter_teleop_remote(self) -> None:
        pass

    async def on_exit_operation(self) -> None:
        pass

    async def _on_controller_changed_cmd(self, envelope: dict[str, Any]) -> None:
        if envelope.get("command") == "status":
            return
        # Backend sends `controller` at the top level of the envelope, not under "data".
        controller = envelope.get("controller")
        policy_uuid: str | None = None
        if isinstance(controller, dict):
            policy_uuid = (
                controller.get("uuid")
                or controller.get("id")
                or controller.get("controller_policy_uuid")
            )
        new_key = policy_uuid if policy_uuid else None
        if self._controller_initialized and new_key == self._last_controller_uuid:
            return
        self._controller_initialized = True
        self._last_controller_uuid = new_key
        if not isinstance(controller, dict):
            await self._set_operation_mode(DriverOperationMode.NO_OP)
            self._emit_driver_info(controller_policy_uuid=None, controller_type=None)
            await self.on_controller_removed()
        else:
            ctype = (
                controller.get("controller_type") or controller.get("type") or ""
            ).strip().lower()
            if ctype == "localop":
                await self._set_operation_mode(DriverOperationMode.TELEOP_LOCAL)
            else:
                await self._set_operation_mode(DriverOperationMode.TELEOP_REMOTE)
            self._emit_driver_info(controller_policy_uuid=policy_uuid, controller_type=ctype)
            await self.on_controller_assigned(ctype, policy_uuid)

    async def on_controller_assigned(self, ctype: str, policy_uuid: str | None) -> None:
        """Called when a controller is assigned. Override to add driver-specific alerts."""
        logger.info("Controller assigned (type=%r, uuid=%r)", ctype, policy_uuid)

    async def on_controller_removed(self) -> None:
        """Called when the controller is removed. Override to add driver-specific alerts."""
        logger.info("Controller removed — awaiting new controller assignment")

    async def _on_teleoperate_cmd(self, envelope: dict[str, Any]) -> None:
        if envelope.get("command") == "status":
            return
        await self._set_operation_mode(DriverOperationMode.TELEOP_LOCAL)

    async def _on_remoteoperate_cmd(self, envelope: dict[str, Any]) -> None:
        if envelope.get("command") == "status":
            return
        await self._set_operation_mode(DriverOperationMode.TELEOP_REMOTE)

    async def _on_stop_cmd(self, envelope: dict[str, Any]) -> None:
        if envelope.get("command") == "status":
            return
        await self._set_operation_mode(DriverOperationMode.NO_OP)

    def driver_info_extra(self) -> dict[str, Any]:
        """Subclass hook: extra fields merged into driver_info snapshots."""
        return {}

    def _emit_driver_info(self, **fields: Any) -> None:
        """Queue fields for the registry ``twin/telemetry`` publisher (debounced)."""
        self._telemetry.update(**fields)

    def _build_driver_telemetry_snapshot(self) -> dict[str, Any]:
        from cyberwave.twin.telemetry import TwinTelemetry

        from ..support.utils import get_sdk_version

        lifecycle = getattr(self, "_lifecycle_state", None)
        snapshot = TwinTelemetry.standard_driver_info(
            lifecycle_state=getattr(lifecycle, "value", None) if lifecycle else None,
            operation_mode=self._operation_mode.value,
            driver_family=type(self).driver_family,
            sdk_version=get_sdk_version(),
            registry_id=self._resolved_registry_id(),
        )
        snapshot.update(self.driver_info_extra())
        return snapshot

    def _publish_driver_telemetry_payload(self, payload: dict[str, Any]) -> None:
        twin = getattr(self, "_twin", None)
        if twin is None:
            return
        twin.publish_telemetry(payload)

    def _driver_telemetry_publisher_callback(self) -> None:
        """Registry publisher: flush debounced snapshot via transport (no return payload)."""
        self._telemetry.publish_if_dirty()

    def _emit_driver_info_after_schema(self, schema: dict[str, Any]) -> None:
        commands = schema.get("commands") or {}
        supported = commands.get("supported") or []
        self._emit_driver_info(
            mqtt_schema_version=schema.get("schema_version"),
            commands_count=len(supported) if isinstance(supported, list) else 0,
        )

    async def _wire_interface_from_registry(self) -> None:
        cw = self._require_client()
        twin_uuid = self._twin_uuid_for_wire()
        prefix = self._mqtt_prefix_for_wire()
        mode = self._operation_mode

        cmd_table = self._interface.command_dispatch_table(mode)
        if cmd_table:
            cmd_topic = TopicSpec(
                namespace="twin",
                leaf="command",
                payload_schema_ref="TwinCommandPayload",
            )
            path = resolve_topic_path(cmd_topic, twin_uuid, prefix=prefix)

            async def command_dispatch(envelope: dict[str, Any]) -> None:
                if envelope.get("command") == "status":
                    return
                name = envelope.get("command")
                if not isinstance(name, str):
                    return
                group = cmd_table.get(name)
                if group is None:
                    return
                await self._invoke_listener(group.callback, envelope)

            self._subscribe_mqtt(path, command_dispatch)

        for entry in self._interface.non_command_listeners_for_mode(mode):
            m = mqtt_spec(entry.topic)
            if m is None:
                continue
            path = resolve_topic_path(m, twin_uuid, prefix=prefix)
            allowed = entry.protocol.source_types if entry.protocol else None
            cb = filtered_listener(entry.callbacks.callback, allowed)

            async def topic_handler(
                envelope: dict[str, Any], _cb: Any = cb
            ) -> None:
                await self._invoke_listener(_cb, envelope)

            no_local = (
                entry.topic.namespace == "joint" and entry.topic.leaf == "update"
            )
            self._subscribe_mqtt(path, topic_handler, no_local=no_local)

        await self._wire_zenoh_from_registry()

        logger.info(
            "Wired interface registry (mode=%s, commands=%d, listeners=%d)",
            mode.value,
            len(cmd_table),
            len(self._interface.non_command_listeners_for_mode(mode)),
        )

    async def _unwire_interface_from_registry(self) -> None:
        await self._teardown_zenoh_registry()
        cw = getattr(self, "_cw", None)
        if cw is None:
            # Cloud connect never completed (e.g. startup failed before MQTT).
            # Nothing was wired, so there is nothing to unsubscribe. Requiring a
            # client here would raise and mask the original startup exception in
            # run_async's finally block.
            self._wired_mqtt_handlers.clear()
            return
        mqtt = cw.mqtt
        for path, _handler in self._wired_mqtt_handlers:
            try:
                if hasattr(mqtt, "unsubscribe"):
                    mqtt.unsubscribe(path)
            except Exception:
                logger.debug("unsubscribe failed for %s", path, exc_info=True)
        self._wired_mqtt_handlers.clear()

    async def _run_registry_publishers(self) -> None:
        if not self._is_active_for_publishers():
            return
        twin_uuid = self._twin_uuid_for_wire()
        prefix = self._mqtt_prefix_for_wire()
        cw = self._require_client()
        self._publisher_tick_counter += 1
        for entry in self._interface.tick_publishers_for_mode(self._operation_mode):
            rate = entry.publisher.rate_hz
            if rate is not None and rate > 0:
                tick_hz = getattr(type(self), "TICK_RATE_HZ", 10.0)
                if self._publisher_tick_counter % max(1, int(round(tick_hz / rate))) != 0:
                    continue
            cb = entry.callbacks.callback
            if cb is None:
                continue
            payload = await self._invoke_publisher(cb)
            if payload is None:
                continue
            m = mqtt_spec(entry.topic)
            if m is not None and entry.publish_mode in {"mqtt", "dual"}:
                path = resolve_topic_path(m, twin_uuid, prefix=prefix)
                cw.mqtt.publish(path, payload)
            z = zenoh_spec(entry.topic)
            if z is not None and entry.publish_mode in {"zenoh", "dual"}:
                if z.wire_format == "ndarray":
                    logger.debug(
                        "Skipping registry Zenoh tick publish on %s (ndarray); "
                        "use zenoh_publish_frame in device callbacks",
                        z.channel,
                    )
                else:
                    self._registry_zenoh_publish(z.channel, payload)

    @staticmethod
    async def _invoke_listener(
        callback: Any, envelope: dict[str, Any]
    ) -> None:
        result = callback(envelope)
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _adapt_async_mqtt_handler(
        handler: ListenerHandler,
        *,
        loop: asyncio.AbstractEventLoop,
        path: str,
    ) -> Callable[[dict[str, Any]], None]:
        """Bridge Paho's sync callback thread to the driver's asyncio loop."""

        def sync_wrapper(envelope: dict[str, Any]) -> None:
            future = asyncio.run_coroutine_threadsafe(handler(envelope), loop)

            def _log_failure(done: asyncio.Future[None]) -> None:
                try:
                    done.result()
                except Exception:
                    logger.exception("MQTT async handler failed on %s", path)

            future.add_done_callback(_log_failure)

        return sync_wrapper

    @staticmethod
    async def _invoke_publisher(callback: Any) -> dict[str, Any] | None:
        result = callback()
        if inspect.isawaitable(result):
            result = await result
        if result is None:
            return None
        if not isinstance(result, dict):
            if result is not None:
                logger.warning("publisher callback returned non-dict: %r", type(result))
            return None
        return result

    def _subscribe_mqtt(
        self,
        path: str,
        handler: ListenerHandler,
        *,
        no_local: bool = False,
    ) -> None:
        cw = self._require_client()
        mqtt_handler: ListenerHandler = handler
        if inspect.iscoroutinefunction(handler):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.error(
                    "Cannot wire async MQTT handler on %s: no running event loop",
                    path,
                )
            else:
                mqtt_handler = self._adapt_async_mqtt_handler(
                    handler, loop=loop, path=path
                )
        cw.mqtt.subscribe(path, mqtt_handler, no_local=no_local)
        if no_local and getattr(cw.mqtt, "is_mqtt_v5", False):
            logger.info(
                "MQTT v5 no_local subscribe on %s (edge publish echo filtered)",
                path,
            )
        elif no_local:
            logger.info(
                "Subscribed to %s (set CYBERWAVE_MQTT_PROTOCOL=5 for no_local echo filter)",
                path,
            )
        self._wired_mqtt_handlers.append((path, mqtt_handler))

    def _require_client(self) -> Any:
        cw = getattr(self, "_cw", None)
        if cw is None:
            raise RuntimeError(
                "Cyberwave client not connected — run the driver lifecycle first"
            )
        return cw

    def _require_twin(self) -> Any:
        twin = getattr(self, "_twin", None)
        if twin is None:
            raise RuntimeError("twin not available")
        return twin

    def _twin_uuid_for_wire(self) -> str:
        return str(getattr(self, "twin_uuid"))

    def _mqtt_prefix_for_wire(self) -> str:
        cw = getattr(self, "_cw", None)
        if cw is None:
            return ""
        return str(getattr(cw.mqtt, "topic_prefix", "") or "")

    def _is_interface_wired(self) -> bool:
        return bool(self._wired_mqtt_handlers)

    def _is_active_for_publishers(self) -> bool:
        state = getattr(self, "_lifecycle_state", None)
        return getattr(state, "value", None) == "active"

    def wire_interface_from_registry_sync(self) -> None:
        """Synchronous wrapper for ROS 2 lifecycle nodes."""
        asyncio.run(self._wire_interface_from_registry())

    def unwire_interface_from_registry_sync(self) -> None:
        asyncio.run(self._unwire_interface_from_registry())

    def run_registry_publishers_sync(self) -> None:
        asyncio.run(self._run_registry_publishers())

    def _registry_zenoh_requested(self) -> bool:
        """True when define_interface registered Zenoh or dual topic specs."""
        return self._interface.has_zenoh_publishers() or self._interface.has_zenoh_subscribers()

    def _registry_zenoh_enabled(self) -> bool:
        """Zenoh is driven by the interface registry, not CYBERWAVE_DATA_BACKEND."""
        if not self._registry_zenoh_requested():
            return False
        try:
            from cyberwave.data.config import is_zenoh_publish_enabled

            return bool(is_zenoh_publish_enabled())
        except Exception:
            return True

    def _ensure_registry_zenoh_bus(self) -> None:
        if self._registry_zenoh_bus is not None:
            return
        if not self._registry_zenoh_enabled():
            return
        try:
            cw = self._require_client()
            twin_uuid = self._twin_uuid_for_wire()
            self._registry_zenoh_bus = cw.data_bus_for(twin_uuid)
            logger.info(
                "[ZENOH] Registry data bus active (twin=%s)",
                twin_uuid,
            )
        except Exception as exc:
            logger.warning("[ZENOH] Registry bus init failed: %s", exc)

    def _registry_zenoh_publish(self, channel: str, payload: Any) -> None:
        if self._registry_zenoh_bus is None:
            return
        try:
            self._registry_zenoh_bus.publish(channel, payload)
        except Exception as exc:
            logger.warning("[ZENOH] Registry publish failed on %s: %s", channel, exc)

    async def _wire_zenoh_from_registry(self) -> None:
        if not self._registry_zenoh_enabled():
            return
        if not self._interface.has_zenoh_subscribers():
            return
        self._ensure_registry_zenoh_bus()
        if self._registry_zenoh_bus is None:
            return

        mode = self._operation_mode
        loop = asyncio.get_running_loop()

        for entry in self._interface.zenoh_subscribe_entries_for_mode(mode):
            z = zenoh_spec(entry.topic)
            if z is None:
                continue
            channel = z.channel
            cb = entry.callbacks.callback

            def _on_decoded(
                decoded: Any,
                _cb: Any = cb,
                _channel: str = channel,
            ) -> None:
                if not isinstance(decoded, dict):
                    return
                if self._registry_zenoh_teleop_blocks_commands():
                    return
                asyncio.run_coroutine_threadsafe(
                    self._invoke_listener(_cb, decoded), loop
                )

            sub = self._registry_zenoh_bus.subscribe(channel, _on_decoded)
            self._registry_zenoh_subscriptions.append(sub)

        for entry in self._interface.zenoh_command_entries_for_mode(mode):
            z = zenoh_spec(entry.topic)
            if z is None:
                continue
            channel = z.channel
            cb = entry.callbacks.callback
            watchdog_ms = z.watchdog_ms

            def _on_command(
                decoded: Any,
                _cb: Any = cb,
                _channel: str = channel,
            ) -> None:
                if not isinstance(decoded, dict):
                    return
                if self._registry_zenoh_teleop_blocks_commands():
                    return
                asyncio.run_coroutine_threadsafe(
                    self._invoke_listener(_cb, decoded), loop
                )

            sub = self._registry_zenoh_bus.subscribe(channel, _on_command)
            self._registry_zenoh_subscriptions.append(sub)
            if watchdog_ms > 0:
                logger.debug(
                    "[ZENOH] commands/%s watchdog_ms=%d (registry listener)",
                    channel,
                    watchdog_ms,
                )

        if self._registry_zenoh_subscriptions:
            logger.info(
                "Wired %d Zenoh registry subscription(s) (mode=%s)",
                len(self._registry_zenoh_subscriptions),
                mode.value,
            )

    def _registry_zenoh_teleop_blocks_commands(self) -> bool:
        return self._operation_mode in {
            DriverOperationMode.TELEOP_LOCAL,
            DriverOperationMode.TELEOP_REMOTE,
        }

    async def _teardown_zenoh_registry(self) -> None:
        import contextlib

        for sub in self._registry_zenoh_subscriptions:
            with contextlib.suppress(Exception):
                sub.close()
        self._registry_zenoh_subscriptions.clear()
        self._registry_zenoh_bus = None

    async def _activate_registry_zenoh(self) -> None:
        """Open the registry Zenoh bus after driver activation when publishers exist."""
        if self._interface.has_zenoh_publishers():
            self._ensure_registry_zenoh_bus()

    async def _derive_initial_operation_mode(self) -> None:
        """Set operation mode from the twin's attached controller policy at startup."""
        from ..base import refresh_driver_twin_from_api, resolve_twin_attached_controller

        if self._twin is None:
            return
        self._twin = refresh_driver_twin_from_api(self)
        policy_uuid, ctype = resolve_twin_attached_controller(self._twin)
        if not policy_uuid:
            await self._set_operation_mode(DriverOperationMode.NO_OP)
            return
        self._controller_initialized = True
        self._last_controller_uuid = policy_uuid
        ctype_norm = (ctype or "").strip().lower()
        if ctype_norm == "localop":
            await self._set_operation_mode(DriverOperationMode.TELEOP_LOCAL)
        else:
            await self._set_operation_mode(DriverOperationMode.TELEOP_REMOTE)
        await self.on_controller_assigned(ctype_norm, policy_uuid)
