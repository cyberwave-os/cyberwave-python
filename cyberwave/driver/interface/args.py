"""Topic specs and publisher/listener args for the Python SDK driver registry.

Use with :class:`~cyberwave.driver.interface.registry.DriverInterfaceRegistry`
inside :meth:`~cyberwave.driver.interface.registry_mixin.InterfaceRegistryMixin.define_interface`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal

ListenerCallback = Callable[[dict[str, Any]], None | Awaitable[None]]
PublisherCallback = Callable[[], dict[str, Any] | Awaitable[dict[str, Any]]]

PublishMode = Literal["mqtt", "zenoh", "dual"]
WireFormat = Literal["json", "ndarray"]


class DriverOperationMode(str, Enum):
    """What the robot is allowed to do (orthogonal to :class:`DriverLifecycleState`)."""

    NO_OP = "no_op"
    TELEOP_LOCAL = "teleop_local"
    TELEOP_REMOTE = "teleop_remote"


@dataclass(frozen=True)
class TopicSpec:
    """Cyber topic identity with optional MQTT and/or Zenoh transport."""

    payload_schema_ref: str
    description: str | None = None
    namespace: str | None = None
    leaf: str | None = None
    topic_slug: str | None = None
    enable_mqtt: bool = True
    enable_zenoh: bool = False
    zenoh_channel: str | None = None
    wire_format: WireFormat = "json"
    watchdog_ms: int = 0
    rate_hz: float | None = None
    """Per-publisher outbound Cyber publish cap (Hz). Overrides manifest
    ``mqtt_max_hz`` and the ``STREAM_PUBLISH_MAX_HZ`` class default. Applied by
    the ``from_ros`` forwarder; None → inherit the resolved default. Must be
    positive when set (0 does NOT mean "off" — it would disable throttling)."""

    def __post_init__(self) -> None:
        if not self.enable_mqtt and not self.enable_zenoh:
            raise ValueError("TopicSpec requires enable_mqtt and/or enable_zenoh")
        if self.rate_hz is not None and self.rate_hz <= 0:
            raise ValueError(
                f"TopicSpec.rate_hz must be positive when set (got {self.rate_hz!r}); "
                "omit it to inherit the resolved default cap"
            )
        if self.enable_mqtt:
            has_ns_leaf = self.namespace is not None and self.leaf is not None
            has_slug = self.topic_slug is not None
            if has_ns_leaf == has_slug:
                raise ValueError(
                    "TopicSpec requires exactly one of (namespace+leaf) or topic_slug "
                    "when enable_mqtt is True"
                )
        if self.enable_zenoh:
            from .topic_spec_pairing import forbid_zenoh_on_topic

            forbid_zenoh_on_topic(self)

    def registry_key(self) -> tuple[str, str]:
        if self.topic_slug is not None:
            return ("slug", self.topic_slug)
        assert self.namespace is not None and self.leaf is not None
        return ("ns", f"{self.namespace}/{self.leaf}")

    def resolved_zenoh_channel(self) -> str:
        from .topic_spec_pairing import resolve_zenoh_channel

        return resolve_zenoh_channel(self)


@dataclass(frozen=True)
class ZenohTransportView:
    """Resolved Zenoh transport fields for registry wiring (internal)."""

    channel: str
    payload_schema_ref: str
    description: str | None
    wire_format: WireFormat
    watchdog_ms: int


@dataclass(frozen=True)
class CallbackGroup:
    """Runtime callback (stripped from manifest export).

    For ROS-fed publishers (``from_ros`` on :meth:`~cyberwave.driver.interface.registry.DriverInterfaceRegistry.add_publisher`),
    ``callback=None`` uses the default :func:`~cyberwave.driver.ros2.message_payload.ros_message_to_transport_payload`.
    """

    callback: ListenerCallback | PublisherCallback | None = None
    label: str | None = None


@dataclass(frozen=True)
class ProtocolArgs:
    """Optional MQTT topic metadata (``source_types``, ``units``, …).

    A true value object: list/dict inputs are coerced to immutable forms in
    ``__post_init__`` so an instance can be safely reused across registrations
    and used as a dict/set key. ``units`` reads back as a mapping (``dict(...)``)
    but cannot be mutated through the instance.
    """

    source_types: tuple[str, ...] | None = None
    units: Mapping[str, str] | None = None
    direction_notes: str | None = None
    related_topics: tuple[str, ...] | None = None
    command_input: bool | None = field(default=None, kw_only=True)
    """Whether the device accepts commands on this topic, independent of direction wording."""

    def __post_init__(self) -> None:
        if self.command_input is not None and type(self.command_input) is not bool:
            raise ValueError("command_input must be a boolean when declared")
        if self.source_types is not None:
            object.__setattr__(self, "source_types", tuple(self.source_types))
        if self.related_topics is not None:
            object.__setattr__(self, "related_topics", tuple(self.related_topics))
        if self.units is not None:
            object.__setattr__(self, "units", MappingProxyType(dict(self.units)))


@dataclass(frozen=True)
class CommandArg:
    """One declared command argument; typed fields are compiled by the backend.

    The original three positional fields and their untyped wire shape remain
    supported. Set ``type`` to opt into complete, typed parameter collection
    instead of legacy numeric direction expansion.
    """

    name: str
    default: Any = None
    unit: str | None = None
    type: (
        Literal["string", "number", "integer", "boolean", "array", "object"] | None
    ) = field(default=None, kw_only=True)
    required: bool | None = field(default=None, kw_only=True)
    enum: tuple[str | int | float | bool, ...] | None = field(
        default=None, kw_only=True
    )
    minimum: int | float | None = field(default=None, kw_only=True)
    maximum: int | float | None = field(default=None, kw_only=True)
    min_items: int | None = field(default=None, kw_only=True)
    description: str | None = field(default=None, kw_only=True)

    def to_catalog_dict(self) -> dict[str, Any]:
        """Export the authoring shape consumed by the canonical server compiler."""
        optional = {
            "type": self.type,
            "required": self.required,
            "enum": list(self.enum) if self.enum is not None else None,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "minItems": self.min_items,
            "description": self.description,
        }
        legacy = {"name": self.name, "default": self.default, "unit": self.unit}
        if self.type is None:
            if any(value is not None for value in optional.values()):
                raise ValueError(f"{self.name}: typed argument fields require type")
            return legacy
        # None remains the legacy constructor's omitted-default sentinel. Typed
        # null is not a supported argument type; do not manufacture a null value
        # for a required field that the user still needs to configure.
        return {
            key: value
            for key, value in {**legacy, **optional}.items()
            if value is not None
        }


@dataclass(frozen=True)
class CommandArgs:
    """Command catalog entry for ``twin/command`` dispatch."""

    name: str
    continuous: bool = False
    rate_hz: float | None = None
    default_duration_s: float | None = None
    event_only: bool = False
    # When True, the command listener is still registered and dispatched, but the
    # command is omitted from the exported catalog (``commands.supported``) so it
    # is not advertised as a robot capability. Used for internal management
    # commands (controller-changed, teleoperate, …).
    catalog_hidden: bool = False
    # Human-readable summary of what the command does, surfaced to agents/MCP via
    # the compiled catalog (``commands.specs[name].description``). Defaults to "".
    description: str = ""
    args: tuple[CommandArg, ...] = ()
    # At least one declared typed argument must be supplied; validated by the server compiler.
    required_any_of: tuple[str, ...] | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class PublisherArgs:
    """Publisher scheduling and transport hints."""

    rate_hz: float | None = None
    publish_mode: PublishMode | None = None


# Modes where management commands are always active.
MANAGEMENT_MODES: frozenset[DriverOperationMode] = frozenset(
    {DriverOperationMode.NO_OP}
)

# Default: actuation listeners/publishers require teleop.
TELEOP_MODES: frozenset[DriverOperationMode] = frozenset(
    {
        DriverOperationMode.TELEOP_LOCAL,
        DriverOperationMode.TELEOP_REMOTE,
    }
)


def default_operation_modes(
    *, management: bool = False
) -> frozenset[DriverOperationMode]:
    if management:
        return frozenset(DriverOperationMode)
    return TELEOP_MODES


def mqtt_spec(topic: TopicSpec) -> TopicSpec | None:
    if topic.enable_mqtt:
        return topic
    return None


def zenoh_spec(topic: TopicSpec) -> ZenohTransportView | None:
    if not topic.enable_zenoh:
        return None
    return ZenohTransportView(
        channel=topic.resolved_zenoh_channel(),
        payload_schema_ref=topic.payload_schema_ref,
        description=topic.description,
        wire_format=topic.wire_format,
        watchdog_ms=topic.watchdog_ms,
    )


def inferred_publish_mode(topic: TopicSpec) -> PublishMode:
    if topic.enable_mqtt and topic.enable_zenoh:
        return "dual"
    if topic.enable_zenoh:
        return "zenoh"
    return "mqtt"


def effective_publish_mode(topic: TopicSpec, publisher: PublisherArgs) -> PublishMode:
    mode = publisher.publish_mode or inferred_publish_mode(topic)
    inferred = inferred_publish_mode(topic)
    if publisher.publish_mode is not None and publisher.publish_mode != inferred:
        raise ValueError(
            f"publish_mode={publisher.publish_mode!r} conflicts with TopicSpec "
            f"(enable_mqtt={topic.enable_mqtt}, enable_zenoh={topic.enable_zenoh})"
        )
    return mode
