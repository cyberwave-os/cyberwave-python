"""Declarative MQTT/Zenoh interface registry for Python SDK edge drivers.

Collects :class:`~cyberwave.driver.interface.args.TopicSpec` entries and exports
uncompiled ``cw-driver.yml`` root dicts (compilation happens on the backend).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cyberwave.manifest.driver_config import TWIN_COMMAND_TOPIC_SLUG

MQTT_BUNDLE_SCHEMA_VERSION = 1
ZENOH_BUNDLE_SCHEMA_VERSION = 1

_NAMESPACE_PREFIXES: dict[str, str] = {
    "joint": "cyberwave/joint/{twin_uuid}",
    "twin": "cyberwave/twin/{twin_uuid}",
    "webrtc": "cyberwave/twin/{twin_uuid}",
    "pose": "cyberwave/twin/{twin_uuid}",
    "environment": "cyberwave/environment/{environment_uuid}",
}

_WEBRTC_LEAF_ALIASES: dict[str, str] = {
    "offer": "webrtc-offer",
    "answer": "webrtc-answer",
    "candidate": "webrtc-candidate",
}


def _leaf_to_slug(namespace: str, leaf: str) -> str:
    prefix = _NAMESPACE_PREFIXES.get(namespace)
    if prefix is None:
        raise ValueError(f"Unknown mqtt namespace {namespace!r}")
    if namespace == "webrtc":
        segment = _WEBRTC_LEAF_ALIASES.get(leaf, leaf.replace("_", "-"))
        return f"{prefix}/{segment}"
    if leaf == "+":
        return f"{prefix}/+"
    return f"{prefix}/{leaf}"

from .args import (
    CallbackGroup,
    CommandArgs,
    DriverOperationMode,
    ProtocolArgs,
    PublisherArgs,
    TopicSpec,
    default_operation_modes,
    effective_publish_mode,
    mqtt_spec,
    zenoh_spec,
)

_TOPIC_SLUG_BY_NS_LEAF: dict[tuple[str, str], str] = {}


def _load_slug_map() -> dict[tuple[str, str], str]:
    global _TOPIC_SLUG_BY_NS_LEAF
    if _TOPIC_SLUG_BY_NS_LEAF:
        return _TOPIC_SLUG_BY_NS_LEAF
    from cyberwave.manifest.driver_config import (
        JOINT_UPDATE_TOPIC_SLUG,
        TWIN_COMMAND_TOPIC_SLUG,
        TWIN_TELEMETRY_TOPIC_SLUG,
    )

    _TOPIC_SLUG_BY_NS_LEAF = {
        ("twin", "command"): TWIN_COMMAND_TOPIC_SLUG,
        ("twin", "telemetry"): TWIN_TELEMETRY_TOPIC_SLUG,
        ("joint", "update"): JOINT_UPDATE_TOPIC_SLUG,
    }
    return _TOPIC_SLUG_BY_NS_LEAF


def resolve_topic_path(
    topic: TopicSpec,
    twin_uuid: str,
    *,
    prefix: str = "",
) -> str:
    """Resolve a :class:`TopicSpec` to a concrete MQTT topic string."""
    if topic.topic_slug is not None:
        slug = topic.topic_slug.replace("{twin_uuid}", twin_uuid)
    else:
        assert topic.namespace is not None and topic.leaf is not None
        template = _load_slug_map().get((topic.namespace, topic.leaf))
        if template is None:
            slug = _leaf_to_slug(topic.namespace, topic.leaf)
        else:
            slug = template.replace("{twin_uuid}", twin_uuid)
    return f"{prefix}{slug}"


def _merge_source_types(meta: dict[str, Any], source_types: Any) -> None:
    """Union *source_types* into ``meta`` preserving first-seen order (no clobber).

    A topic registered as both a listener (e.g. ``tele``/``edit``/``sim_tele``)
    and a publisher (``edge``) must advertise *both* sets in the manifest.
    """
    merged = list(meta.get("source_types", []))
    for s in source_types:
        if s not in merged:
            merged.append(s)
    meta["source_types"] = merged


def _merge_direction(meta: dict[str, Any], *, subscribe: bool, publish: bool) -> None:
    """Fold a subscribe/publish registration into ``meta["direction"]``.

    A topic seen as both a listener and a publisher resolves to ``"both"``;
    otherwise it takes the direction of the current registration.
    """
    cur = meta.get("direction", "subscribe")
    if subscribe and publish:
        meta["direction"] = "both"
    elif subscribe and cur == "publish":
        meta["direction"] = "both"
    elif publish and cur == "subscribe":
        meta["direction"] = "both"
    elif publish:
        meta["direction"] = "publish"
    elif subscribe:
        meta["direction"] = "subscribe"


def _slug_for_mqtt(topic: TopicSpec) -> str:
    if topic.topic_slug is not None:
        return topic.topic_slug
    assert topic.namespace is not None and topic.leaf is not None
    return _leaf_to_slug(topic.namespace, topic.leaf)


@dataclass
class _ListenerEntry:
    topic: TopicSpec
    callbacks: CallbackGroup
    protocol: ProtocolArgs | None
    command: CommandArgs | None
    operation_modes: frozenset[DriverOperationMode]


@dataclass
class _PublisherEntry:
    topic: TopicSpec
    callbacks: CallbackGroup
    protocol: ProtocolArgs | None
    publisher: PublisherArgs
    operation_modes: frozenset[DriverOperationMode]
    publish_mode: str
    from_ros: Any | None = None


class DriverInterfaceRegistry:
    """Collect Cyber interface declarations and compile to cw-driver / catalog bundles.

    ``add_listener`` — subscribe on **Cyber** (MQTT). ``add_publisher`` — publish on
    **Cyber** (MQTT/Zenoh). Optional ``from_ros=Ros2TopicSpec`` on publishers reads a
    ROS topic and forwards to Cyber (ROS 2 drivers only).
    """

    def __init__(self) -> None:
        self._listeners: list[_ListenerEntry] = []
        self._publishers: list[_PublisherEntry] = []
        self._topic_meta: dict[str, dict[str, Any]] = {}
        self._zenoh_meta: dict[str, dict[str, Any]] = {}

    def add_listener(
        self,
        topic: TopicSpec,
        callbacks: CallbackGroup,
        *,
        protocol: ProtocolArgs | None = None,
        command: CommandArgs | None = None,
        operation_modes: frozenset[DriverOperationMode] | None = None,
    ) -> None:
        from ..ros2.topic_spec import Ros2TopicSpec

        if isinstance(topic, Ros2TopicSpec):
            raise TypeError(
                "add_listener expects a Cyber TopicSpec; "
                "Ros2TopicSpec belongs on add_publisher(..., from_ros=...)"
            )
        if zenoh_spec(topic) is not None and mqtt_spec(topic) is None:
            raise ValueError(
                "Zenoh-only listeners are not supported; set enable_mqtt=True on TopicSpec"
            )
        if command is not None and not command.name.strip():
            raise ValueError("CommandArgs.name must not be empty")
        if command is not None:
            # Commands (incl. event_only) are available in every operation mode
            # unless the caller scopes them explicitly.
            modes = operation_modes or frozenset(DriverOperationMode)
        else:
            modes = operation_modes or default_operation_modes()
        self._listeners.append(
            _ListenerEntry(
                topic=topic,
                callbacks=callbacks,
                protocol=protocol,
                command=command,
                operation_modes=modes,
            )
        )
        m = mqtt_spec(topic)
        if m is not None:
            self._merge_topic_meta(m, protocol, subscribe=True)
        if topic.enable_zenoh:
            self._merge_zenoh_meta(topic, protocol, subscribe=True)

    def add_publisher(
        self,
        topic: TopicSpec,
        callbacks: CallbackGroup,
        *,
        from_ros: Any | None = None,
        protocol: ProtocolArgs | None = None,
        publisher: PublisherArgs | None = None,
        operation_modes: frozenset[DriverOperationMode] | None = None,
    ) -> None:
        if from_ros is not None:
            from ..ros2.topic_spec import Ros2TopicSpec

            if not isinstance(from_ros, Ros2TopicSpec):
                raise TypeError(
                    "from_ros must be a Ros2TopicSpec (Cyber publishers only on topic=)"
                )
        pub = publisher or PublisherArgs()
        if from_ros is not None and pub.rate_hz is not None:
            raise ValueError(
                "PublisherArgs.rate_hz cannot be set when from_ros is used "
                "(ROS message rate drives publishing)"
            )
        mode = effective_publish_mode(topic, pub)
        self._publishers.append(
            _PublisherEntry(
                topic=topic,
                callbacks=callbacks,
                protocol=protocol,
                publisher=pub,
                operation_modes=operation_modes or default_operation_modes(),
                publish_mode=mode,
                from_ros=from_ros,
            )
        )
        m = mqtt_spec(topic)
        if m is not None and mode in {"mqtt", "dual"}:
            self._merge_topic_meta(m, protocol, publish=True)
        if topic.enable_zenoh and mode in {"zenoh", "dual"}:
            self._merge_zenoh_meta(topic, protocol, publish=True)

    def _merge_topic_meta(
        self,
        topic: TopicSpec,
        protocol: ProtocolArgs | None,
        *,
        subscribe: bool = False,
        publish: bool = False,
    ) -> None:
        slug = _slug_for_mqtt(topic)
        meta = self._topic_meta.setdefault(
            slug,
            {
                "payload_schema_ref": topic.payload_schema_ref,
                "description": topic.description or "",
                "direction": "subscribe",
            },
        )
        if topic.description and not meta.get("description"):
            meta["description"] = topic.description
        if protocol:
            if protocol.source_types:
                _merge_source_types(meta, protocol.source_types)
            if protocol.units:
                meta.setdefault("units", {}).update(dict(protocol.units))
            if protocol.direction_notes:
                meta["direction_notes"] = protocol.direction_notes
        _merge_direction(meta, subscribe=subscribe, publish=publish)

    def _merge_zenoh_meta(
        self,
        topic: TopicSpec,
        protocol: ProtocolArgs | None,
        *,
        subscribe: bool = False,
        publish: bool = False,
    ) -> None:
        channel = topic.resolved_zenoh_channel()
        meta = self._zenoh_meta.setdefault(
            channel,
            {
                "payload_schema_ref": topic.payload_schema_ref,
                "description": topic.description or "",
                "direction": "subscribe",
                "wire_format": topic.wire_format,
            },
        )
        if topic.description and not meta.get("description"):
            meta["description"] = topic.description
        if topic.watchdog_ms > 0:
            meta["watchdog_ms"] = topic.watchdog_ms
        if protocol:
            if protocol.source_types:
                _merge_source_types(meta, protocol.source_types)
            if protocol.units:
                meta.setdefault("units", {}).update(dict(protocol.units))
            if protocol.direction_notes:
                meta["direction_notes"] = protocol.direction_notes
        _merge_direction(meta, subscribe=subscribe, publish=publish)

    def listeners_for_mode(
        self, mode: DriverOperationMode
    ) -> list[_ListenerEntry]:
        return [e for e in self._listeners if mode in e.operation_modes]

    def publishers_for_mode(
        self, mode: DriverOperationMode
    ) -> list[_PublisherEntry]:
        return [e for e in self._publishers if mode in e.operation_modes]

    def ros_forward_publishers_for_mode(
        self, mode: DriverOperationMode
    ) -> list[_PublisherEntry]:
        """Publishers that read from ROS and forward to Cyber (``from_ros`` set)."""
        return [
            e
            for e in self.publishers_for_mode(mode)
            if e.from_ros is not None
        ]

    def tick_publishers_for_mode(
        self, mode: DriverOperationMode
    ) -> list[_PublisherEntry]:
        """Publishers driven by the driver tick loop (no ``from_ros``)."""
        return [
            e
            for e in self.publishers_for_mode(mode)
            if e.from_ros is None
        ]

    def command_dispatch_table(
        self, mode: DriverOperationMode
    ) -> dict[str, CallbackGroup]:
        table: dict[str, CallbackGroup] = {}
        for entry in self.listeners_for_mode(mode):
            if entry.command is None:
                continue
            name = entry.command.name
            if name in table:
                raise ValueError(f"duplicate command registration: {name!r}")
            table[name] = entry.callbacks
        return table

    def non_command_listeners_for_mode(
        self, mode: DriverOperationMode
    ) -> list[_ListenerEntry]:
        return [
            e
            for e in self.listeners_for_mode(mode)
            if e.command is None
            and (
                mqtt_spec(e.topic) is None
                or _slug_for_mqtt(mqtt_spec(e.topic)) != TWIN_COMMAND_TOPIC_SLUG
            )
        ]

    def zenoh_subscribe_entries_for_mode(
        self, mode: DriverOperationMode
    ) -> list[_ListenerEntry]:
        result: list[_ListenerEntry] = []
        for entry in self.listeners_for_mode(mode):
            z = zenoh_spec(entry.topic)
            if z is None or entry.command is not None or z.channel.startswith("commands/"):
                continue
            result.append(entry)
        return result

    def zenoh_command_entries_for_mode(
        self, mode: DriverOperationMode
    ) -> list[_ListenerEntry]:
        result: list[_ListenerEntry] = []
        for entry in self.listeners_for_mode(mode):
            z = zenoh_spec(entry.topic)
            if z is None or not z.channel.startswith("commands/"):
                continue
            result.append(entry)
        return result

    def has_zenoh_publishers(self) -> bool:
        return any(
            e.publish_mode in {"zenoh", "dual"} and zenoh_spec(e.topic) is not None
            for e in self._publishers
        )

    def has_zenoh_subscribers(self) -> bool:
        return any(zenoh_spec(e.topic) is not None for e in self._listeners)

    def to_cw_driver_dict(
        self,
        *,
        registry_id: str,
        driver_family: str = "python",
        schema_version: int = MQTT_BUNDLE_SCHEMA_VERSION,
        constraints: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build cw-driver.yml root dict (no callbacks)."""
        mqtt: dict[str, Any] = {
            "schema_version": schema_version,
            "driver_family": driver_family,
        }
        for slug, entry in self._topic_meta.items():
            if slug == TWIN_COMMAND_TOPIC_SLUG:
                ns, leaf = "twin", "command"
            elif slug.startswith("cyberwave/joint/{twin_uuid}/"):
                ns, leaf = "joint", slug.split("/")[-1]
            elif slug.startswith("cyberwave/twin/{twin_uuid}/"):
                ns, leaf = "twin", slug.split("/")[-1]
            else:
                continue
            mqtt.setdefault(ns, {})[leaf] = dict(entry)

        commands_supported: list[Any] = []
        seen_cmds: set[str] = set()
        for entry in self._listeners:
            if entry.command is None:
                continue
            cmd = entry.command
            if cmd.catalog_hidden:
                continue  # registered + dispatched, but not advertised in the catalog
            if cmd.name in seen_cmds:
                continue
            seen_cmds.add(cmd.name)
            if (
                cmd.continuous
                or cmd.rate_hz is not None
                or cmd.default_duration_s is not None
                or cmd.args
            ):
                item: dict[str, Any] = {"name": cmd.name}
                if cmd.continuous:
                    item["continuous"] = True
                if cmd.rate_hz is not None:
                    item["rate_hz"] = cmd.rate_hz
                if cmd.default_duration_s is not None:
                    item["default_duration_s"] = cmd.default_duration_s
                if cmd.args:
                    item["args"] = [
                        {"name": a.name, "default": a.default, "unit": a.unit}
                        for a in cmd.args
                    ]
                commands_supported.append(item)
            else:
                commands_supported.append(cmd.name)

        if commands_supported:
            mqtt["commands"] = {"supported": commands_supported}
        if constraints:
            mqtt["constraints"] = list(constraints)

        ros_forward: list[dict[str, Any]] = []
        for entry in self._publishers:
            if entry.from_ros is None:
                continue
            mqtt_m = mqtt_spec(entry.topic)
            item: dict[str, Any] = {
                "ros_topic": entry.from_ros.topic,
            }
            if mqtt_m is not None:
                item["mqtt_topic_slug"] = _slug_for_mqtt(mqtt_m)
                item["payload_schema_ref"] = mqtt_m.payload_schema_ref
            z = zenoh_spec(entry.topic)
            if z is not None:
                item["zenoh_channel"] = z.channel
            ros_forward.append(item)

        result: dict[str, Any] = {
            "registry_id": registry_id,
            "mqtt": mqtt,
        }
        if ros_forward:
            result["ros2"] = {"forward_publishers": ros_forward}
        if self._zenoh_meta:
            result["zenoh"] = {
                "schema_version": ZENOH_BUNDLE_SCHEMA_VERSION,
                "driver_family": driver_family,
                "channels": {k: dict(v) for k, v in self._zenoh_meta.items()},
            }
        return result


def default_management_commands(
    registry: DriverInterfaceRegistry,
    *,
    on_controller_changed: CallbackGroup,
    on_teleoperate: CallbackGroup | None = None,
    on_remoteoperate: CallbackGroup | None = None,
    on_stop: CallbackGroup | None = None,
    catalog_hidden: bool = False,
) -> None:
    """Register SO101-style management commands on ``twin/command``.

    ``catalog_hidden`` keeps the listeners functional but omits these management
    commands from the exported catalog (``commands.supported``) — useful for
    drivers (e.g. Piper) that should not advertise controller-changed /
    teleoperate / remoteoperate / stop as robot capabilities.
    """
    cmd_topic = TopicSpec(
        namespace="twin",
        leaf="command",
        payload_schema_ref="TwinCommandPayload",
        description="Command ingress and management events",
    )
    all_modes = frozenset(DriverOperationMode)
    registry.add_listener(
        cmd_topic,
        on_controller_changed,
        command=CommandArgs(
            name="controller-changed", event_only=True, catalog_hidden=catalog_hidden
        ),
        operation_modes=all_modes,
    )
    if on_teleoperate is not None:
        registry.add_listener(
            cmd_topic,
            on_teleoperate,
            command=CommandArgs(name="teleoperate", catalog_hidden=catalog_hidden),
            operation_modes=all_modes,
        )
    if on_remoteoperate is not None:
        registry.add_listener(
            cmd_topic,
            on_remoteoperate,
            command=CommandArgs(name="remoteoperate", catalog_hidden=catalog_hidden),
            operation_modes=all_modes,
        )
    if on_stop is not None:
        registry.add_listener(
            cmd_topic,
            on_stop,
            command=CommandArgs(name="stop", catalog_hidden=catalog_hidden),
            operation_modes=all_modes,
        )
